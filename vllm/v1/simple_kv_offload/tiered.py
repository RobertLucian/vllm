# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler side of TieredCPUOffloadConnector: a RAM tier that extends VRAM.

VRAM -> RAM: the GPU prefix cache evicts least recently used first, except
the blocks queued requests resume from, which both tiers keep until nothing
else may go (`observe_queue`): LRU would take them first, as their requests
wait. The lazy walk of `SimpleCPUOffloadScheduler` copies every block a
lookup can use before the GPU may drop it, including the blocks of exact-end
and prompt-tail states (keys that end inside a block), each with the other
groups' blocks under the same keys. With `copy_ahead_blocks` the walk keeps
that many blocks at the eviction end copied, so an allocation (a restore's
most of all) rarely waits for a copy. Recurrent states at block boundaries
are copied only where live sessions resume (`_keep_resume_points`: the last
boundaries before the latest turn's prompt end and end, and before the prompt
that began its tool loop), at points several continuations share (branch
points, and shared prefixes' junctions: `_note_junction`), and every
`ram_state_spacing_blocks` blocks if set: a state takes a page of RAM per
Mamba group, as much as the attention pages of a whole block. A turn that
starts a tool loop keeps the state before its prompt end from its admission
(`_hold_prefill_points`): the GPU would otherwise drop it as the turn runs
past it. When a turn supersedes the one it resumed, the states only the older
turn resumed at leave both tiers, and blocks only retired sessions owned are
never copied (`RamGreedyDualSizeFrequency.disowned`). With `prune_gpu_states`
the GPU also releases a finished request's other states, instead of holding
them until its LRU order reaches them.

RAM -> VRAM ("promote, then hit"): before a waiting request's next lookup,
the connector finds the deepest point the request can resume at with blocks
from either tier, loads the blocks only RAM holds into GPU blocks, caches them
under their own keys (re-indexing an exact-end state as a GPU frontier), and
pins them until the request is admitted. The ordinary GPU lookup then makes
the hit; no tokens are ever loaded as an external suffix, so a session split
across the tiers in any order resumes whole. Restores run side by side while
a step's blocks stay free for the running requests (`_start_restore`). A
request waits for room to restore into as long as blocks take to free up;
the requests behind it in the queue pass it for `_HEAD_PASS_S`, then wait
with it (`_behind_head`), so a long session's restore is not starved by
shorter ones: resuming from the GPU alone would recompute what RAM holds.
"""

import bisect
import itertools
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.core.hybrid_checkpoint.cost import PrefillCostModel
from vllm.v1.core.hybrid_checkpoint.frontier import _SUFFIX
from vllm.v1.core.hybrid_checkpoint.keys import frontier_key
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import MambaSpec, SlidingWindowSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.metrics import SimpleCPUOffloadStats, TieredMetricName

from vllm.v1.simple_kv_offload.ram_policy import (
    RamEvictionStats,
    RamGreedyDualSizeFrequency,
)

if TYPE_CHECKING:
    from vllm.v1.core.hybrid_checkpoint.manager import HybridCheckpointManager
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

RAM_EVICTION_POLICIES = ("lru", "gdsf")

# A restore is worth a load only when it resumes this many tokens deeper than
# the GPU alone.
_MIN_RESTORE_GAIN_TOKENS = 64
# Restores admitted no later than this pin their blocks; then the pins go.
# With a prefill cost model the pin timeout grows with the prefill the
# restore saves (`_RESTORE_WAIT_SHARE` of it), up to `_MAX_PIN_TIMEOUT_S`: a
# 120k-token session takes minutes to prefill again.
_PIN_TIMEOUT_S = 30.0
_MAX_PIN_TIMEOUT_S = 120.0
_RESTORE_WAIT_SHARE = 0.5
# A request waits for room to restore into for as long as its session is in
# RAM: resuming from the GPU alone recomputes what RAM holds, in the same GPU
# blocks the restore waits for. This bounds the wait only if blocks never
# free up. The requests behind it in the queue pass it for `_HEAD_PASS_S`
# (shorter restores and GPU-only resumes run meanwhile), then wait with it
# until it is admitted (`_behind_head`). Holding them from the start
# serialized restores: on real GPUs the tiered connector ran 26-54% of the
# eager mirror's turns at equal pools (tests/v1/hybrid_gpu).
_MAX_RESTORE_WAIT_S = 600.0
_HEAD_PASS_S = 5.0
# Steps between lookups of what a queued request resumes from
# (`observe_queue`): a running request may compute a deeper point it can
# resume at.
_REQUEUE_STEPS = 32
# The share of both pools, GPU and RAM, that queued requests' blocks are kept
# in, from the queue's head (`observe_queue`): a queue longer than that would
# leave only the sessions of requests about to come back to give.
_QUEUE_KEEP_SHARE = 0.5
# The share of RAM their keys may hold: with RAM about the size of VRAM, the
# share of both pools is all of RAM, and RAM would make room for the next
# copies (queued blocks the GPU moves down among them) from the queue's own.
# With RAM over half again as large as VRAM it never binds: a lower share
# kept fewer of pi-length sessions' requests (their keys are mostly in RAM)
# and cost turns.
_QUEUE_RAM_SHARE = 0.75
# Finished requests' ends remembered to measure resumes against, the sessions
# whose resume points the tiers keep, and shared prefixes' junctions.
_MAX_SEEN_ENDS = 65536
# Steps a state stays pending for pruning while a request still holds it.
_PRUNE_RETRY_STEPS = 64
# Upper bounds of the recompute buckets (tokens recomputed that a resume had
# already seen).
RECOMPUTE_BUCKETS = (0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536)


def _end(request: "Request") -> int:
    """The tokens a finished request processed. Under async scheduling its
    computed count also holds the slots of the step still in flight (the
    sampled token and the drafts after it), past the tokens it has."""
    return min(request.num_computed_tokens, len(request.all_token_ids) - 1)


def pin_timeout_s(
    cost: PrefillCostModel | None, gpu_only_tokens: int, num_tokens: int
) -> float:
    """How long a restore's blocks stay pinned for its request's admission:
    a share of the prefill it saves over resuming at `gpu_only_tokens`
    (`_RESTORE_WAIT_SHARE`), within bounds."""
    if cost is None:
        return _PIN_TIMEOUT_S
    saved = cost.seconds(gpu_only_tokens, num_tokens)
    return min(_MAX_PIN_TIMEOUT_S, max(_PIN_TIMEOUT_S, _RESTORE_WAIT_SHARE * saved))


@dataclass
class RestoreStats:
    """Cumulative counters of RAM -> VRAM restores."""

    # Restores started and completed, and the blocks and exact-end states
    # they loaded.
    started: int = 0
    completed: int = 0
    restored_blocks: int = 0
    restored_frontiers: int = 0
    # Steps a restore waited for the spill guard, or for the pin cap.
    waited_guard: int = 0
    waited_cap: int = 0
    # Steps a request waited behind a restore that waits for room.
    waited_behind: int = 0
    # Requests that resumed from the GPU alone after waiting too long.
    gave_up: int = 0
    # Restores whose request was not admitted before the pins timed out.
    pin_timeouts: int = 0
    # Restores whose request left before the load finished.
    orphaned: int = 0
    # Times a waiting restore gave up its RAM pins: the spill guard had no
    # other RAM block to copy into (`_relieve_spill_stall`).
    unpinned_for_spill: int = 0


RESTORE_EVENT_TO_FIELD = {f: f for f in RestoreStats.__dataclass_fields__}


@dataclass(slots=True)
class _Unit:
    """An exact-end state whose blocks the RAM tier holds, every group's."""

    key: BlockHash
    anchor: BlockHash | None
    num_tokens: int
    suffix: tuple[int, ...]
    groups: tuple[int, ...]


class _EndIndex:
    """Exact ends by (anchor, length, suffix), as `FrontierIndex` keys them:
    the RAM tier's exact-end states, and the ends finished requests reached."""

    def __init__(self) -> None:
        self.entries: dict[BlockHash, _Unit] = {}
        self.index: dict[
            BlockHash | None, dict[int, dict[tuple[int, ...], list[BlockHash]]]
        ] = {}

    def add(self, unit: _Unit) -> None:
        self.entries[unit.key] = unit
        by_length = self.index.setdefault(unit.anchor, {}).setdefault(
            unit.num_tokens, {}
        )
        by_length.setdefault(unit.suffix, []).append(unit.key)

    def remove(self, key: BlockHash) -> _Unit | None:
        unit = self.entries.pop(key, None)
        if unit is None:
            return None
        lengths = self.index[unit.anchor]
        suffixes = lengths[unit.num_tokens]
        suffixes[unit.suffix].remove(key)
        if not suffixes[unit.suffix]:
            del suffixes[unit.suffix]
            if not suffixes:
                del lengths[unit.num_tokens]
                if not lengths:
                    del self.index[unit.anchor]
        return unit

    def clear(self) -> None:
        self.entries.clear()
        self.index.clear()


@dataclass(slots=True)
class _Piece:
    """A block on a resume path, and whether the GPU holds it."""

    block: KVCacheBlock
    on_gpu: bool


@dataclass(slots=True)
class _Plan:
    num_tokens: int
    pieces: list[_Piece]
    # The exact-end state the RAM tier restores whole, if the point is one.
    frontier: _Unit | None = None
    # Mamba state keys (and the length they hold) the point resumes at.
    state: tuple[BlockHash, int] | None = None
    # Where the request would resume from the GPU alone.
    gpu_only_tokens: int = 0

    @property
    def on_gpu_only(self) -> bool:
        return all(p.on_gpu for p in self.pieces)


@dataclass(slots=True)
class _Restore:
    request_id: str
    num_tokens: int
    # (GPU block, RAM block) to load.
    loads: list[tuple[KVCacheBlock, KVCacheBlock]]
    # Path blocks the GPU already holds, pinned with the loaded ones.
    pinned: list[KVCacheBlock]
    frontier: _Unit | None
    state: tuple[BlockHash, int] | None
    pin_timeout_s: float = _PIN_TIMEOUT_S
    event: int = -1
    done: bool = False
    done_at: float = 0.0
    orphaned: bool = False
    reset: bool = False

    @property
    def num_pinned(self) -> int:
        return len(self.pinned) + len(self.loads)


@dataclass
class _ResumeStats:
    """Resumes by tier: count, recomputed seen tokens, cumulative buckets."""

    counts: dict[str, int] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    buckets: dict[tuple[str, str], int] = field(default_factory=dict)

    def record(self, tier: str, recomputed: int) -> None:
        self.counts[tier] = self.counts.get(tier, 0) + 1
        self.tokens[tier] = self.tokens.get(tier, 0) + recomputed
        start = bisect.bisect_left(RECOMPUTE_BUCKETS, recomputed)
        for bound in (*RECOMPUTE_BUCKETS[start:], None):
            le = "+Inf" if bound is None else str(bound)
            self.buckets[tier, le] = self.buckets.get((tier, le), 0) + 1


class TieredCPUOffloadScheduler(SimpleCPUOffloadScheduler):
    """Scheduler-side manager of TieredCPUOffloadConnector (module docstring)."""

    def __init__(
        self,
        *args: Any,
        ram_eviction: str = "lru",
        prefill_cost_coefficients: tuple[float, ...] | None = None,
        frequency_half_life_s: float = 600.0,
        ram_state_spacing_blocks: int = 0,
        prune_gpu_states: bool = True,
        copy_ahead_blocks: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs["lazy_offload"] = True
        super().__init__(*args, **kwargs)
        if ram_eviction not in RAM_EVICTION_POLICIES:
            raise ValueError(
                f"ram_eviction must be one of {RAM_EVICTION_POLICIES}, "
                f"got {ram_eviction!r}"
            )
        if ram_state_spacing_blocks < 0:
            raise ValueError("ram_state_spacing_blocks must be 0 (off) or more")
        if copy_ahead_blocks < 0:
            raise ValueError("copy_ahead_blocks must be 0 (the walk's own) or more")
        # The lazy walk covers at least this many blocks at the eviction end.
        self._target_free = max(self._target_free, copy_ahead_blocks)
        self.ram_eviction = ram_eviction
        self.prefill_cost_coefficients = prefill_cost_coefficients
        self._cost = (
            PrefillCostModel(*prefill_cost_coefficients[:2])
            if prefill_cost_coefficients and len(prefill_cost_coefficients) >= 2
            else None
        )
        self.frequency_half_life_s = frequency_half_life_s
        groups = self.cpu_kv_cache_config.kv_cache_groups
        self.mamba_gids = [
            g
            for g in self.prefix_cacheable_group_ids
            if isinstance(groups[g].kv_cache_spec, MambaSpec)
        ]
        self._mamba_gid_set = set(self.mamba_gids)
        # Sliding-window groups, and the blocks a cache hit reads from each
        # before its point (its window): a resume restores them too, at block
        # boundaries only (`_window_pieces`).
        self._window_blocks = {
            g: cdiv(spec.sliding_window - 1, self.block_size)
            for g in self.prefix_cacheable_group_ids
            if isinstance(spec := groups[g].kv_cache_spec, SlidingWindowSpec)
        }
        # A window in blocks of another size than the pages' is not restored.
        self._windows_restorable = all(
            self.group_block_sizes[g] == self.block_size for g in self._window_blocks
        )
        self.units_per_block = self.block_size // self.hash_block_size
        self._state_spacing = ram_state_spacing_blocks * self.block_size
        # Block-boundary states both tiers keep besides the spaced ones: each
        # live session's resume points (`_keep_resume_points`), counted by the
        # sessions that keep them; by session, where its tool loop began and
        # its points.
        self._kept: Counter[BlockHash] = Counter()
        self._kept_by: OrderedDict[BlockHash, tuple[int, tuple[BlockHash, ...]]] = (
            OrderedDict()
        )
        # By running request, the points its session will keep that its
        # prefill computes, and the hash units RAM keeps for it meanwhile
        # (`_hold_prefill_points`).
        self._held_points: dict[str, tuple[BlockHash, ...]] = {}
        self._held_units: dict[str, tuple[BlockHash, ...]] = {}
        # Resume points no live session keeps any more. One stays a point
        # several continuations share only if resumed more than once: its
        # other continuation is usually its own session's past, the turns a
        # user prompt rendered anew.
        self._superseded: OrderedDict[BlockHash, None] = OrderedDict()
        # Shared prefixes' junctions: block boundaries a request found pages
        # at that another request wrote, with no state to resume from
        # (`_note_junction`). The most recently found or resumed kept.
        self._junctions: OrderedDict[BlockHash, None] = OrderedDict()
        # Requests admitted extending the end their session reached (a tool
        # loop's next turn), not only its prompt.
        self._extends_end: set[str] = set()
        # The GPU keeps the same block-boundary states of a finished request's
        # chain; the others go when it finishes (`_prune_gpu_states`).
        self.prune_gpu_states = prune_gpu_states
        # Finished chains' state keys, pruned once their requests' blocks are
        # free (the scheduler frees them after `request_finished`); the steps
        # a key held by another request has waited.
        self._prune_pending: list[BlockHash] = []
        self._prune_waits: dict[BlockHash, int] = {}
        self.prune_outcomes: Counter[str] = Counter()
        self._prune_outcomes_snapshot: Counter[str] = Counter()
        self.num_gpu_states_pruned = 0
        self._gpu_states_pruned_snapshot = 0
        self.num_ram_states_dropped = 0
        self._ram_states_dropped_snapshot = 0
        self._kv_cache_manager: KVCacheManager | None = None
        # Exact-end states in RAM, and the ends finished requests reached:
        # their ends and their prompts' (the session each belongs to).
        self.ram_frontiers = _EndIndex()
        self._seen_ends = _EndIndex()
        self._seen_order: OrderedDict[BlockHash, None] = OrderedDict()
        self._session_of: dict[BlockHash, BlockHash] = {}
        # Each end's tokens past its last hash unit; and by the hash unit
        # before that one, its tokens past it (`_seen_end`).
        self._seen_tails: dict[BlockHash, tuple[int, ...]] = {}
        self._seen_by_previous: dict[
            BlockHash | None, dict[BlockHash, tuple[int, ...]]
        ] = {}
        self._previous_of: dict[BlockHash, BlockHash | None] = {}
        # Restores: by request, queued for the next load event, by event.
        self._restores: dict[str, _Restore] = {}
        self._unsent: list[_Restore] = []
        self._restores_by_event: dict[int, list[_Restore]] = {}
        self._load_event_pending_counts: dict[int, int] = {}
        self._loading_cpu: set[int] = set()
        self._num_pinned = 0
        # Request -> when it first waited for room to restore into. The first
        # of them in the queue (its id and queue position), and the RAM blocks
        # it would load, pinned so that making room does not evict them.
        self._waiting_since: dict[str, float] = {}
        self._head: tuple[str, tuple[int, float, str]] | None = None
        self._head_pins: list[KVCacheBlock] = []
        # By queued request, the step its keys were found at and the keys of
        # the blocks it resumes from, shallowest first (`observe_queue`).
        self._queued: dict[str, tuple[int, tuple[BlockHashWithGroupId, ...]]] = {}
        # The units of the kept requests' keys, and of their points' states:
        # the GPU neither drops nor prunes them uncopied, nor RAM its copies.
        self._queued_units: set[BlockHash] = set()
        self._queued_states: set[BlockHash] = set()
        self._queue_steps = 0
        self._priority_policy = self.vllm_config.scheduler_config.policy == "priority"
        self._observed: set[str] = set()
        # Admitted at a point a restore loaded, until `observe_admission`.
        self._admitted_from_ram: set[str] = set()
        self.restore_stats = RestoreStats()
        self._restore_snapshot = RestoreStats()
        self.resume_stats = _ResumeStats()
        self._resume_snapshot = _ResumeStats()
        self.cpu_block_pool.key_removal_listener = self._ram_keys_removed
        # Request -> the session end it resumed, which its own session
        # inherits the frequency of.
        self._resumed_from: dict[str, BlockHash] = {}
        self.ram_policy: RamGreedyDualSizeFrequency | None = None
        if ram_eviction == "gdsf":
            if not prefill_cost_coefficients or len(prefill_cost_coefficients) < 2:
                raise ValueError(
                    "ram_eviction='gdsf' needs prefill_cost_coefficients [D, E]"
                )
            assert self._cost is not None
            self.ram_policy = RamGreedyDualSizeFrequency(
                self.cpu_block_pool,
                cost=self._cost,
                page_group_id=self.fa_gidx,
                state_group_ids=[
                    g for g in self.prefix_cacheable_group_ids if g != self.fa_gidx
                ],
                block_size=self.block_size,
                half_life_s=frequency_half_life_s,
                on_gpu=self._on_gpu,
                gpu_keeps=self._gpu_keeps,
                leaving=self.leaving_copies,
                dropped=self.drop_leaving,
            )
            self.cpu_block_pool.eviction_policy = self.ram_policy
            # The policy ranks RAM; a GPU eviction does not refresh it.
            self.refresh_cpu_on_gpu_eviction = False
        self._ram_evictions_snapshot = RamEvictionStats()

    # ------------------------------------------------------------------
    # Binding

    def bind_kv_cache_manager(self, kv_cache_manager: "KVCacheManager") -> None:
        self._kv_cache_manager = kv_cache_manager

    @property
    def _checkpoints(self) -> "HybridCheckpointManager | None":
        manager = self._kv_cache_manager
        return None if manager is None else manager.checkpoints

    # ------------------------------------------------------------------
    # VRAM -> RAM

    def _on_gpu(self, key: BlockHashWithGroupId) -> bool:
        pool = self._gpu_block_pool
        return (
            pool is not None
            and pool.cached_block_hash_to_block.get_one_block(key) is not None
        )

    def _gpu_keeps(self) -> Callable[[BlockHashWithGroupId], bool]:
        """Whether the GPU holds the block under a key and will not drop it
        soon: a request holds it, or it is cached outside the eviction end the
        lazy walk copies ahead (`_leave_window` blocks). RAM's copy of such a
        block only mirrors it (`RamGreedyDualSizeFrequency._take_duplicates`);
        the walk copies the block again once it reaches that end."""
        pool = self._gpu_block_pool
        if pool is None:
            return lambda key: False
        policy = pool.eviction_policy
        window = self._leave_window
        order = (
            policy.peek_eviction_order(window)
            if policy is not None
            else pool.free_block_queue.iter_blocks_after(None)
        )
        leaving = {b.block_id for b in itertools.islice(order, window)}
        cached = pool.cached_block_hash_to_block

        def keeps(key: BlockHashWithGroupId) -> bool:
            block = cached.get_one_block(key)
            return block is not None and (
                block.ref_cnt > 0 or block.block_id not in leaving
            )

        return keeps

    def _restorable(self, block: KVCacheBlock) -> bool:
        """Every key a lookup can use is restored, including keys that end
        inside a block (exact-end and prompt-tail states), except states
        between resume points (module docstring) and what only retired
        sessions owned (`RamGreedyDualSizeFrequency.disowned`)."""
        return (
            block.block_hash is not None
            and not self._thinned(block)
            and not self._disowned(block)
        )

    def _disowned(self, block: KVCacheBlock) -> bool:
        policy, pool = self.ram_policy, self._gpu_block_pool
        if policy is None or pool is None:
            return False
        assert block.block_hash is not None
        extra = pool.cached_block_hashes_by_block.get(block.block_id, ())
        keys = (block.block_hash, *extra)
        # A queued request's (a sibling of a superseded turn) is read again.
        if any(get_block_hash(key) in self._queued_units for key in keys):
            return False
        return policy.disowned(keys)

    def _spaced(self, num_tokens: int) -> bool:
        """Whether both tiers keep the block-boundary state at `num_tokens`
        whatever resumes there (`ram_state_spacing_blocks`)."""
        return bool(self._state_spacing) and not num_tokens % self._state_spacing

    def _thinned(self, block: KVCacheBlock) -> bool:
        """Whether `block` is a block-boundary state RAM does not keep: not a
        spaced one, not a live session's resume point or a queued request's,
        and not a point several continuations share."""
        assert block.block_hash is not None
        num_tokens = block.block_hash_num_tokens
        if (
            get_group_id(block.block_hash) not in self._mamba_gid_set
            or num_tokens is None
            or num_tokens % self.block_size
            or self._spaced(num_tokens)
        ):
            return False
        checkpoints = self._checkpoints
        pool = self._gpu_block_pool
        if (
            checkpoints is None
            or pool is None
            # Another key (an exact-end state at a boundary).
            or pool.cached_block_hashes_by_block.get(block.block_id)
        ):
            return False
        block_hash = get_block_hash(block.block_hash)
        if block_hash in self._kept or block_hash in self._queued_states:
            return False
        return not self._shared(block_hash)

    def _shared(self, key: BlockHash) -> bool:
        """Whether the state at `key` is a point several continuations share:
        a shared prefix's junction, or a branch point of the checkpoint graph
        (`_superseded`). The graph alone does not keep a junction: the states
        a prefill caches are linked only once its request finishes, and a
        unit rebuilt from RAM starts without hits."""
        if key in self._junctions:
            return True
        checkpoints = self._checkpoints
        unit = None if checkpoints is None else checkpoints.graph.units.get(key)
        return (
            unit is not None
            and unit.is_branch_point
            and (unit.hits > 1 or key not in self._superseded)
        )

    def needs_spill(self, block: KVCacheBlock) -> bool:
        """SpillGuard: also a block of an exact-end state whose other blocks
        RAM does not hold yet. The GPU drops such a state whole, so a block
        copied ahead of the others leaving first would take them along
        uncopied. Not a block whose copy RAM dropped by value as it left
        (`drop_leaving`)."""
        if block.block_hash in self._dropped_keys:
            return False
        return self._uncopied(block) or (
            not block.ref_cnt and self._frontier_incomplete(block)
        )

    def _frontier_incomplete(self, block: KVCacheBlock) -> bool:
        checkpoints = self._checkpoints
        if checkpoints is None or block.block_hash is None or block.is_null:
            return False
        frontiers = checkpoints.frontiers
        if not frontiers.entries:
            return False
        pool = self._gpu_block_pool
        assert pool is not None
        cpu = self.cpu_block_pool.cached_block_hash_to_block
        policy = self.ram_policy
        for key in (
            block.block_hash,
            *pool.cached_block_hashes_by_block.get(block.block_id, ()),
        ):
            entry = frontiers.entries.get(get_block_hash(key))
            if entry is None:
                continue
            if policy is not None and policy.disowned(
                make_block_hash_with_group_id(entry.key, g) for g in entry.blocks
            ):
                # A state only retired sessions resumed at: never copied.
                continue
            for group_id in entry.blocks:
                if (
                    cpu.get_one_block(
                        make_block_hash_with_group_id(entry.key, group_id)
                    )
                    is None
                ):
                    return True
        return False

    def _boundary_siblings(self, block: KVCacheBlock) -> list[KVCacheBlock]:
        """The other groups' blocks under any of `block`'s keys: an exact-end
        state's attention page carries the state's key beside its own."""
        assert block.block_hash is not None and self._gpu_block_pool is not None
        pool = self._gpu_block_pool
        gpu = pool.cached_block_hash_to_block
        cpu = self.cpu_block_pool.cached_block_hash_to_block
        keys = (
            block.block_hash,
            *pool.cached_block_hashes_by_block.get(block.block_id, ()),
        )
        siblings: dict[int, KVCacheBlock] = {}
        for own in keys:
            group_id = get_group_id(own)
            if self.group_block_sizes[group_id] != self.fa_block_size:
                continue
            block_hash = get_block_hash(own)
            for sibling_group in self.prefix_cacheable_group_ids:
                if (
                    sibling_group == group_id
                    or self.group_block_sizes[sibling_group] != self.fa_block_size
                ):
                    continue
                key = make_block_hash_with_group_id(block_hash, sibling_group)
                sibling = gpu.get_one_block(key)
                if (
                    sibling is None
                    or sibling.block_id == block.block_id
                    or sibling.block_id in self._unpinned_stores
                    or (sibling.ref_cnt and sibling_group != self.fa_gidx)
                    or cpu.get_one_block(key) is not None
                    or not self._restorable(sibling)
                ):
                    continue
                siblings[sibling.block_id] = sibling
        return list(siblings.values())

    def _process_store_completion(
        self,
        gpu_block_ids: list[int],
        cpu_block_ids: list[int],
        block_meta: Any = None,
    ) -> None:
        """Index the exact-end states whose blocks are now all in RAM."""
        frontiers = None if self._checkpoints is None else self._checkpoints.frontiers
        stored = []
        if frontiers is not None:
            pool = self._gpu_block_pool
            assert pool is not None
            for block_id in gpu_block_ids:
                block = pool.blocks[block_id]
                for key in (
                    block.block_hash,
                    *pool.cached_block_hashes_by_block.get(block_id, ()),
                ):
                    if key is None:
                        continue
                    entry = frontiers.entries.get(get_block_hash(key))
                    if entry is not None:
                        stored.append(entry)
        super()._process_store_completion(gpu_block_ids, cpu_block_ids, block_meta)
        if self.ram_policy is not None:
            cpu_pool = self.cpu_block_pool
            keys: list[BlockHashWithGroupId] = []
            for block_id in cpu_block_ids:
                block = cpu_pool.blocks[block_id]
                if block.block_hash is not None:
                    keys.append(block.block_hash)
                    keys.extend(cpu_pool.cached_block_hashes_by_block.get(block_id, ()))
            self.ram_policy.stored(keys, cpu_block_ids)
        cpu = self.cpu_block_pool.cached_block_hash_to_block
        for entry in stored:
            if entry.key in self.ram_frontiers.entries:
                continue
            groups = tuple(entry.blocks)
            if all(
                cpu.get_one_block(make_block_hash_with_group_id(entry.key, g))
                is not None
                for g in groups
            ):
                self.ram_frontiers.add(
                    _Unit(
                        entry.key, entry.anchor, entry.num_tokens, entry.suffix, groups
                    )
                )

    def _ram_keys_removed(self, keys: list[BlockHashWithGroupId]) -> None:
        """An exact-end state the RAM tier no longer holds whole is dropped:
        its other blocks lose its key too."""
        cpu_pool = self.cpu_block_pool
        cpu = cpu_pool.cached_block_hash_to_block
        for grouped in keys:
            unit = self.ram_frontiers.remove(get_block_hash(grouped))
            if unit is None:
                continue
            for group_id in unit.groups:
                key = make_block_hash_with_group_id(unit.key, group_id)
                block = cpu.get_one_block(key)
                if block is None or block.ref_cnt:
                    continue
                if block.block_hash != key:
                    cpu_pool.uncache_key(block, key)
                else:
                    cpu_pool._maybe_evict_cached_block(block)

    # ------------------------------------------------------------------
    # RAM -> VRAM

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Never an external hit: (None, False) while a restore for the
        request is pending or the request waits behind one (the scheduler
        retries it next step), else (0, False)."""
        request_id = request.request_id
        restore = self._restores.get(request_id)
        if restore is not None:
            if not restore.done:
                return None, False
            self._raise_junction(request, num_computed_tokens)
            return 0, False
        if self._gpu_block_pool is None:
            return 0, False
        if self._behind_head(request):
            self.restore_stats.waited_behind += 1
            return None, False
        if request.skip_reading_external_kv:
            return 0, False
        plan = self._plan(request)
        if plan is None:
            self._stop_waiting(request_id)
            self._raise_junction(request, num_computed_tokens)
            return 0, False
        return self._start_restore(request, plan)

    def _raise_junction(self, request: "Request", num_computed_tokens: int) -> None:
        """The scheduler marks a shared prefix's junction from the GPU alone
        (`Request.shared_prefix_boundary`): where the pages run past the
        point the request resumes at, which another request wrote. With
        pages in RAM too, the junction is where the pages either tier holds
        end: the request's prefill stops there, and the state it caches is
        one both tiers keep (`_note_junction`). Not where a state is cached
        already (the request resumes short of it only because restoring it
        would gain too little)."""
        chain = self._page_chain(request)
        end = len(chain) * self.block_size
        if end <= max(num_computed_tokens, request.shared_prefix_boundary):
            return
        key = request.block_hashes[end // self.hash_block_size - 1]
        if all(
            self._find(make_block_hash_with_group_id(key, group_id))
            for group_id in self.mamba_gids
        ):
            return
        request.shared_prefix_boundary = end

    def _queue_position(self, request: "Request") -> tuple[int, float, str]:
        """Where `request` stands in the scheduler's waiting queue (ties go by
        request id, as `Request.__lt__` orders them)."""
        return (
            request.priority if self._priority_policy else 0,
            request.arrival_time,
            request.request_id,
        )

    def _behind_head(self, request: "Request") -> bool:
        """Whether `request` comes after a request that has waited
        `_HEAD_PASS_S` for room to restore into: it would take the blocks
        that one waits for. Until then it passes it."""
        head = self._head
        if (
            head is None
            or head[0] == request.request_id
            or self._queue_position(request) <= head[1]
        ):
            return False
        since = self._waiting_since.get(head[0])
        return since is not None and time.monotonic() - since >= _HEAD_PASS_S

    def _stop_waiting(self, request_id: str) -> None:
        self._waiting_since.pop(request_id, None)
        if self._head is not None and self._head[0] == request_id:
            self._head = None
            self._unpin_head()

    def _unpin_head(self) -> None:
        if self._head_pins:
            self.cpu_block_pool.free_blocks(self._head_pins)
            self._head_pins = []

    def _loads_in_flight(self) -> bool:
        return bool(self._restores_by_event or self._unsent)

    def _kept_for_queue(self, block: KVCacheBlock) -> bool:
        """Queue protection keeps the block at the GPU's far end
        (`observe_queue`)."""
        units = self._queued_units
        if not units or block.block_hash is None:
            return False
        pool = self._gpu_block_pool
        assert pool is not None
        return any(
            get_block_hash(key) in units
            for key in (
                block.block_hash,
                *pool.cached_block_hashes_by_block.get(block.block_id, ()),
            )
        )

    def _relieve_spill_stall(self) -> bool:
        """The waiting restore's pins on the RAM blocks it would load: RAM
        has no other block to copy the GPU's into, and the restore waits for
        those GPU blocks. RAM may then evict its blocks by value."""
        if not self._head_pins:
            return False
        self._unpin_head()
        self.restore_stats.unpinned_for_spill += 1
        return True

    def observe_queue(self, waiting: Iterable["Request"]) -> None:
        """Keep what the queued requests resume from until they are admitted
        (`_resume_keys`): on the GPU at the far end of the LRU order and in
        RAM out of eviction's reach (`RamGreedyDualSizeFrequency.queued`).
        The queue's requests are the next to read their blocks, yet LRU would
        take those first: they have aged as their requests waited. Requests
        from the queue's head are kept while their blocks fit
        `_QUEUE_KEEP_SHARE` of both pools and `_QUEUE_RAM_SHARE` of RAM; the
        states they resume at are neither pruned nor dropped uncopied
        (`_thinned`, `_disowned`, `_apply_prunes`, `_drop_ram_state`, the
        checkpoint graph's `queued_units`). When nothing else may go, the back
        of the queue goes first, each request's deepest units first. A
        request's keys are found when it joins the queue, again once any of
        them left both tiers, and every `_REQUEUE_STEPS` steps (a running
        request may compute a deeper point it can resume at)."""
        pool = self._gpu_block_pool
        if pool is None:
            return
        self._queue_steps += 1
        steps = self._queue_steps
        # The scheduler's order: by priority, else its own (preempted
        # requests go to the front).
        order = (
            sorted(waiting, key=self._queue_position)
            if self._priority_policy
            else list(waiting)
        )
        gpu = pool.cached_block_hash_to_block
        cpu = self.cpu_block_pool.cached_block_hash_to_block
        previous = self._queued
        queued: dict[str, tuple[int, tuple[BlockHashWithGroupId, ...]]] = {}
        for request in order:
            entry = previous.get(request.request_id)
            if (
                entry is None
                or steps - entry[0] >= _REQUEUE_STEPS
                or any(
                    gpu.get_one_block(key) is None and cpu.get_one_block(key) is None
                    for key in entry[1]
                )
            ):
                entry = (steps, self._resume_keys(request))
            queued[request.request_id] = entry
        self._queued = queued
        policy = self.ram_policy
        if not queued and (policy is None or not policy.has_queued):
            self._queued_units = self._queued_states = set()
            if self._checkpoints is not None:
                self._checkpoints.queued_units = ()
            return
        gpu_queue = pool.free_block_queue
        # A RAM pool in LRU order keeps them at its far end too.
        tiers = (
            ((gpu_queue, gpu),)
            if policy is not None
            else ((gpu_queue, gpu), (self.cpu_block_pool.free_block_queue, cpu))
        )
        num_cpu_blocks = self.cpu_block_pool.num_gpu_blocks
        budget = _QUEUE_KEEP_SHARE * (pool.num_gpu_blocks + num_cpu_blocks)
        ram_budget = _QUEUE_RAM_SHARE * num_cpu_blocks
        kept: list[tuple[BlockHashWithGroupId, ...]] = []
        for request in order:
            keys = queued[request.request_id][1]
            # Counts keys, not blocks: a block can hold several keys, and a
            # key's block can sit in both tiers (RAM copies of what the GPU
            # keeps go first).
            budget -= len(keys)
            ram_budget -= sum(cpu.get_one_block(key) is not None for key in keys)
            if budget < 0 or ram_budget < 0:
                break
            kept.append(keys)
        self._queued_units = {get_block_hash(key) for keys in kept for key in keys}
        self._queued_states = {
            get_block_hash(key)
            for keys in kept
            for key in keys
            if get_group_id(key) in self._mamba_gid_set
        }
        if self._checkpoints is not None:
            self._checkpoints.queued_units = self._queued_states
        # The back of the queue first, each request's deepest first: the
        # queue's head ends at the far end.
        for keys in reversed(kept):
            for key in reversed(keys):
                for queue, cached in tiers:
                    block = cached.get_one_block(key)
                    if block is not None and block.prev_free_block is not None:
                        queue.remove(block)
                        queue.append(block)
        if policy is not None:
            policy.queued(kept)

    def _resume_keys(self, request: "Request") -> tuple[BlockHashWithGroupId, ...]:
        """The keys of the blocks at the deepest point `request` can resume at
        with blocks of either tier, its pages from the first, then the
        point's own (states, an exact end, the page they end in), the last of
        them the deepest; none if it reads no cached KV. Not those of the
        deepest point the GPU alone holds: when that one is close enough to
        resume at instead of restoring (`_plan`), keeping it would cost the
        request the tokens between them."""
        if request.skip_reading_external_kv:
            return ()
        plan = next(self._resume_points(request), None)
        if plan is None:
            return ()
        keys: dict[BlockHashWithGroupId, None] = {}
        for piece in plan.pieces:
            block = piece.block
            if block.block_hash is None:
                continue
            pool = self._gpu_block_pool if piece.on_gpu else self.cpu_block_pool
            assert pool is not None
            keys[block.block_hash] = None
            for key in pool.cached_block_hashes_by_block.get(block.block_id, ()):
                keys[key] = None
        return tuple(keys)

    def _start_restore(
        self, request: "Request", plan: _Plan
    ) -> tuple[int | None, bool]:
        request_id = request.request_id
        pool = self._gpu_block_pool
        assert pool is not None
        loads = [p.block for p in plan.pieces if not p.on_gpu]
        pinned = [p.block for p in plan.pieces if p.on_gpu]
        if any(b.block_id in self._loading_cpu for b in loads):
            # Another request's restore is loading the same RAM blocks.
            return None, False
        num_blocks = len(loads)
        stats = self.restore_stats
        # Restores waiting for admission run side by side while the running
        # requests keep a step's blocks free (the lazy walk's target); a
        # restore alone may pin a whole session. A cap of a quarter of the
        # pool let one 60-120-block session restore at a time in production.
        free = pool.get_num_free_blocks() - sum(not b.ref_cnt for b in pinned)
        if self._num_pinned and free - num_blocks < self._target_free:
            stats.waited_cap += 1
            return self._wait(request, loads)
        # Blocks pinned below leave the free queue: they are not safe to take.
        # Once the spill guard lets held allocations go, neither does this one
        # wait for copies (`KVCacheManager.allocate_slots`).
        safe = (
            pool.num_spill_safe_free_blocks(num_blocks, exclude=pinned)
            if self.holding
            else free
        )
        if safe < num_blocks:
            stats.waited_guard += 1
            # Only what copies can free: blocks the pool lacks altogether come
            # back as requests finish, not from the walk.
            if self.holding and min(num_blocks, free) > safe:
                self.on_allocation_held(
                    min(num_blocks, free) - safe, min(num_blocks, free)
                )
            return self._wait(request, loads)
        self._stop_waiting(request_id)
        pool.touch(pinned)
        gpu_blocks = pool.get_new_blocks(num_blocks)
        self.cpu_block_pool.touch(loads)
        wait_s = self._restore_wait_s(plan)
        restore = _Restore(
            request_id,
            plan.num_tokens,
            list(zip(gpu_blocks, loads)),
            pinned,
            plan.frontier,
            plan.state,
            pin_timeout_s=max(_PIN_TIMEOUT_S, wait_s),
        )
        self._restores[request_id] = restore
        self._unsent.append(restore)
        self._loading_cpu.update(b.block_id for b in loads)
        self._num_pinned += restore.num_pinned
        stats.started += 1
        return None, False

    def _wait(
        self, request: "Request", loads: list[KVCacheBlock]
    ) -> tuple[int | None, bool]:
        """Wait a step for room to restore into (`_MAX_RESTORE_WAIT_S`). The
        first such request in the queue holds the requests behind it back
        and pins the RAM blocks it would load."""
        request_id = request.request_id
        now = time.monotonic()
        since = self._waiting_since.setdefault(request_id, now)
        if now - since >= _MAX_RESTORE_WAIT_S:
            self._stop_waiting(request_id)
            self.restore_stats.gave_up += 1
            return 0, False
        head = self._head
        position = self._queue_position(request)
        if head is None or head[0] == request_id or position < head[1]:
            self._unpin_head()
            self._head = (request_id, position)
            # The GPU blocks it waits for may need copies in RAM first: it
            # pins its own only while the rest of RAM still takes those
            # (copies of blocks about to leave take nothing: `leaving_copies`).
            cpu_pool = self.cpu_block_pool
            free = cpu_pool.get_num_free_blocks() - len(self.leaving_copies())
            if free - len(loads) >= len(loads):
                cpu_pool.touch(loads)
                self._head_pins = list(loads)
        return None, False

    def _plan(self, request: "Request") -> _Plan | None:
        """The deepest point `request` can resume at with blocks from either
        tier, when RAM holds some of them and it resumes deeper than the GPU
        alone; else None."""
        deepest = None
        for plan in self._resume_points(request):
            if deepest is None:
                if plan.on_gpu_only:
                    return None
                deepest = plan
                continue
            if plan.on_gpu_only:
                break
        else:
            plan = None
        if deepest is None:
            return None
        gpu_only = 0 if plan is None else plan.num_tokens
        if deepest.num_tokens - gpu_only < _MIN_RESTORE_GAIN_TOKENS:
            return None
        deepest.gpu_only_tokens = gpu_only
        return deepest

    def _restore_wait_s(self, plan: _Plan) -> float:
        return pin_timeout_s(self._cost, plan.gpu_only_tokens, plan.num_tokens)

    def _find(self, key: BlockHashWithGroupId) -> _Piece | None:
        """The block cached under `key` in either tier, the GPU's first."""
        pool = self._gpu_block_pool
        assert pool is not None
        block = pool.cached_block_hash_to_block.get_one_block(key)
        if block is not None:
            return _Piece(block, True)
        block = self.cpu_block_pool.cached_block_hash_to_block.get_one_block(key)
        # A block another restore is loading counts: the restore waits for
        # that load (`_start_restore`).
        return None if block is None else _Piece(block, False)

    def _page_chain(self, request: "Request") -> list[_Piece]:
        """The full-attention pages of `request` either tier holds, from the
        first, short of its last token."""
        per_block, hashes = self.units_per_block, request.block_hashes
        chain: list[_Piece] = []
        for i in range(
            min(len(hashes) // per_block, (request.num_tokens - 1) // self.block_size)
        ):
            piece = self._find(
                make_block_hash_with_group_id(
                    hashes[(i + 1) * per_block - 1], self.fa_gidx
                )
            )
            if piece is None:
                break
            chain.append(piece)
        return chain

    def _resume_points(self, request: "Request") -> Iterator[_Plan]:
        """Points `request` can resume at with blocks from either tier,
        deepest first: exact-end states (GPU frontiers and RAM ones), then
        states at hash-unit boundaries. With sliding-window groups, only
        block boundaries whose windows either tier holds (`_window_pieces`):
        the GPU's lookup after the restore finds a hit there only with them."""
        windows = self._window_blocks
        if windows and not self._windows_restorable:
            return
        find = self._find
        unit = self.hash_block_size
        hashes = request.block_hashes
        max_length = request.num_tokens - 1
        chain = self._page_chain(request)
        checkpoints = self._checkpoints
        frontiers = None if checkpoints is None else checkpoints.frontiers
        hash_fn = None if frontiers is None else frontiers.hash_fn
        tokens = request.all_token_ids
        # A point past the chain's next block needs a page neither tier holds.
        top = min(
            max_length // unit,
            len(hashes),
            ((len(chain) + 1) * self.block_size - 1) // unit,
        )
        for h in range(top, -1, -1):
            anchor = hashes[h - 1] if h else None
            # An exact end inside a block would need a window's partial page.
            if hash_fn is not None and not windows:
                yield from self._frontier_points(
                    request,
                    h,
                    anchor,
                    max_length,
                    chain,
                    find,
                    frontiers,
                    hash_fn,
                    tokens,
                )
            if not h:
                break
            num_tokens = h * unit
            k = num_tokens // self.block_size
            if k > len(chain) or (windows and num_tokens % self.block_size):
                continue
            block_hash = hashes[h - 1]
            states = []
            for group_id in self.mamba_gids:
                piece = find(make_block_hash_with_group_id(block_hash, group_id))
                if piece is None:
                    break
                states.append(piece)
            else:
                pieces = chain[:k] + states
                if num_tokens % self.block_size:
                    page = find(make_block_hash_with_group_id(block_hash, self.fa_gidx))
                    if page is None:
                        continue
                    pieces.append(page)
                if windows:
                    window = self._window_pieces(hashes, k)
                    if window is None:
                        continue
                    pieces += window
                yield _Plan(num_tokens, pieces, state=(block_hash, num_tokens))

    def _window_pieces(self, hashes: Any, k: int) -> list[_Piece] | None:
        """The sliding-window groups' blocks a cache hit `k` blocks long reads:
        each group's window before it, and one block past it where the
        scheduler drops a speculative decoding block from hits. None if
        neither tier holds one of them."""
        manager = self._kv_cache_manager
        past = 1 if manager is not None and manager.use_eagle else 0
        per_block = self.units_per_block
        last = min(k + past, len(hashes) // per_block)
        pieces = []
        for group_id, window in self._window_blocks.items():
            for i in range(max(0, k - window), last):
                piece = self._find(
                    make_block_hash_with_group_id(
                        hashes[(i + 1) * per_block - 1], group_id
                    )
                )
                if piece is None:
                    return None
                pieces.append(piece)
        return pieces

    def _frontier_points(
        self,
        request: "Request",
        h: int,
        anchor: BlockHash | None,
        max_length: int,
        chain: list[_Piece],
        find: Any,
        frontiers: Any,
        hash_fn: Any,
        tokens: Any,
    ) -> Iterator[_Plan]:
        """Exact-end states `h` hash units in that `request` extends, longest
        first: the GPU's frontiers and the RAM tier's."""
        unit = self.hash_block_size
        candidates: dict[int, list[tuple[bool, Any]]] = {}
        for on_gpu, lengths in (
            (True, frontiers._index.get(anchor)),
            (False, self.ram_frontiers.index.get(anchor)),
        ):
            for num_tokens, by_suffix in (lengths or {}).items():
                if num_tokens > max_length:
                    continue
                start = max(h * unit, num_tokens - _SUFFIX)
                keys = by_suffix.get(tuple(tokens[start:num_tokens]))
                if keys:
                    candidates.setdefault(num_tokens, []).append((on_gpu, keys))
        for num_tokens in sorted(candidates, reverse=True):
            k = num_tokens // self.block_size
            if k > len(chain):
                continue
            expected = frontier_key(hash_fn, request, num_tokens, unit)
            for on_gpu, keys in candidates[num_tokens]:
                if expected not in keys:
                    continue
                if on_gpu:
                    entry = frontiers.entries[expected]
                    if not frontiers._still_cached(entry):
                        continue
                    pieces = [_Piece(b, True) for b in entry.blocks.values()]
                    yield _Plan(num_tokens, chain[:k] + pieces)
                    continue
                ram_unit = self.ram_frontiers.entries[expected]
                pieces = []
                for group_id in ram_unit.groups:
                    piece = find(make_block_hash_with_group_id(expected, group_id))
                    if piece is None or piece.on_gpu:
                        break
                    pieces.append(piece)
                else:
                    yield _Plan(num_tokens, chain[:k] + pieces, frontier=ram_unit)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> SimpleCPUOffloadMetadata:
        meta = super().build_connector_meta(scheduler_output)
        self._apply_prunes()
        self._expire_pins()
        if not self._unsent:
            return meta
        assert meta.load_event < 0, "the tiered connector loads only restores"
        event = self._load_event_counter
        self._load_event_counter += 1
        for restore in self._unsent:
            restore.event = event
            for gpu_block, cpu_block in restore.loads:
                meta.load_gpu_blocks.append(gpu_block.block_id)
                meta.load_cpu_blocks.append(cpu_block.block_id)
        self._restores_by_event[event] = self._unsent
        self._unsent = []
        meta.load_event = event
        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        super().update_connector_output(connector_output)
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, SimpleCPUOffloadWorkerMetadata):
            return
        for event, count in meta.completed_load_events.items():
            total = self._load_event_pending_counts.get(event, 0) + count
            if total >= self._expected_worker_count:
                self._load_event_pending_counts.pop(event, None)
                self._complete(event)
            else:
                self._load_event_pending_counts[event] = total

    def _complete(self, event: int) -> None:
        pool = self._gpu_block_pool
        assert pool is not None
        cpu_pool = self.cpu_block_pool
        for restore in self._restores_by_event.pop(event, []):
            cpu_blocks = [c for _, c in restore.loads]
            self._loading_cpu.difference_update(c.block_id for c in cpu_blocks)
            if not restore.reset:
                for gpu_block, cpu_block in restore.loads:
                    self._adopt_block(gpu_block, cpu_block)
                self._register_units(restore)
                self.restore_stats.completed += 1
                self.restore_stats.restored_blocks += len(restore.loads)
                self._interval_load_blocks_completed += len(restore.loads)
            # The RAM copies were just used: the most recent in RAM.
            cpu_pool.free_blocks(cpu_blocks)
            restore.done = True
            restore.done_at = time.monotonic()
            if restore.orphaned or restore.reset:
                self._release(restore)

    def _adopt_block(self, gpu_block: KVCacheBlock, cpu_block: KVCacheBlock) -> None:
        """Cache the loaded GPU block under every key its RAM copy holds."""
        primary = cpu_block.block_hash
        if primary is None:
            return
        pool = self._gpu_block_pool
        assert pool is not None
        pool._insert_block_hash(
            primary, gpu_block, num_tokens=cpu_block.block_hash_num_tokens
        )
        for key in self.cpu_block_pool.cached_block_hashes_by_block.get(
            cpu_block.block_id, ()
        ):
            pool._insert_block_hash(key, gpu_block, num_tokens=None)

    def _register_units(self, restore: _Restore) -> None:
        """The restored state becomes a unit of the checkpoint graph again,
        and an exact-end state a GPU frontier."""
        checkpoints = self._checkpoints
        if checkpoints is None:
            return
        if restore.state is not None:
            key, num_tokens = restore.state
            if checkpoints.complete(key) and key not in checkpoints.graph.units:
                checkpoints.state_cached(key, num_tokens, None)
        unit = restore.frontier
        if unit is None:
            return
        pool = self._gpu_block_pool
        assert pool is not None
        gpu = pool.cached_block_hash_to_block
        blocks = {}
        for group_id in unit.groups:
            block = gpu.get_one_block(make_block_hash_with_group_id(unit.key, group_id))
            if block is None:
                return
            blocks[group_id] = block
        if checkpoints.frontiers.adopt(
            unit.key, unit.anchor, unit.num_tokens, unit.suffix, blocks
        ):
            checkpoints.state_cached(unit.key, unit.num_tokens, None)
            checkpoints.graph.leaves.add(unit.key)
            self.restore_stats.restored_frontiers += 1

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        restore = self._restores.get(request.request_id)
        if restore is not None and restore.done:
            # The request holds its blocks now.
            self._release(restore)
            self._admitted_from_ram.add(request.request_id)

    def _release(self, restore: _Restore) -> None:
        if self._restores.get(restore.request_id) is restore:
            del self._restores[restore.request_id]
        pool = self._gpu_block_pool
        assert pool is not None
        blocks = restore.pinned + [g for g, _ in restore.loads]
        restore.pinned, restore.loads = [], []
        if blocks:
            pool.free_blocks(blocks)
        self._num_pinned -= len(blocks)

    def _expire_pins(self) -> None:
        now = time.monotonic()
        for restore in list(self._restores.values()):
            if restore.done and now - restore.done_at > restore.pin_timeout_s:
                self.restore_stats.pin_timeouts += 1
                self._release(restore)

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        request_id = request.request_id
        restore = self._restores.get(request_id)
        if restore is not None:
            if restore.done:
                self._release(restore)
            else:
                restore.orphaned = True
                self.restore_stats.orphaned += 1
                del self._restores[request_id]
        self._stop_waiting(request_id)
        self._observed.discard(request_id)
        self._admitted_from_ram.discard(request_id)
        parent = self._resumed_from.pop(request_id, None)
        loop_start = self._loop_start(request, parent)
        key = self._remember_end(request, loop_start)
        if key is not None:
            self._keep_resume_points(request, key, parent, loop_start)
        for state in self._held_points.pop(request_id, ()):
            # Not superseded: no session kept it if this was the last count
            # (the pruning below lets the GPU's copy go).
            self._kept[state] -= 1
            if self._kept[state] <= 0:
                del self._kept[state]
        self._prune_gpu_states(request)
        if self.ram_policy is not None:
            if key is not None:
                self.ram_policy.add_session(
                    key, _end(request), self._session_units(request, loop_start), parent
                )
            self.ram_policy.unhold(self._held_units.pop(request_id, ()))
        return super().request_finished(request, block_ids)

    def _loop_start(self, request: "Request", parent: BlockHash | None) -> int:
        """Where the tool loop a finished request belongs to began: the prompt
        end of the last turn in its session that did not extend the previous
        turn's end (a user prompt; a reasoning model's template drops the
        loop's reasoning then, so the next user prompt parts from the session
        there), or its own prompt end."""
        prompt = min(_end(request), request.num_prompt_tokens)
        extended = request.request_id in self._extends_end
        self._extends_end.discard(request.request_id)
        if extended and parent is not None:
            kept = self._kept_by.get(parent)
            if kept is not None and kept[0] <= prompt:
                return kept[0]
        return prompt

    def _keep_resume_points(
        self,
        request: "Request",
        key: BlockHash,
        parent: BlockHash | None,
        loop_start: int,
    ) -> None:
        """Both tiers keep the states at the last block boundaries before a
        finished request's prompt end, its end and its loop's start: its next
        turn resumes there when the states inside those blocks are gone. The
        session it resumed keeps its own no more: states only that one kept
        leave the GPU and RAM."""
        hashes, per_block = request.block_hashes, self.units_per_block
        end = _end(request)
        points: dict[BlockHash, None] = {}
        for last in (loop_start, min(end, request.num_prompt_tokens), end):
            k = last // self.block_size
            if k and k * per_block <= len(hashes):
                points[hashes[k * per_block - 1]] = None
        # Counted before any are released, so the points they share stay.
        previous = self._kept_by.pop(key, None)
        self._kept_by[key] = (loop_start, tuple(points))
        self._kept.update(list(points))
        if previous is not None:
            self._release_points(previous[1])
        if parent is not None and parent != key:
            self._release_resume_points(parent)
        while len(self._kept_by) > _MAX_SEEN_ENDS:
            self._release_resume_points(next(iter(self._kept_by)))

    def _hold_prefill_points(
        self, request: "Request", parent: BlockHash | None, extends_end: bool
    ) -> None:
        """Both tiers keep the states at the last block boundaries before an
        admitted request's prompt end and its loop's start from now on, not
        only once it finishes (`_keep_resume_points`): the checkpoint graph
        lets a boundary state go as the request runs past it, and the GPU then
        drops it uncopied. Without them its session resumes only near its end,
        and losing that tail re-reads the session from the start."""
        if extends_end and parent in self._kept_by:
            # A tool loop's next turn: its session resumed where the loop
            # began keeps that point, and its own prompt end is superseded
            # once it finishes.
            return
        prompt = request.num_prompt_tokens
        hashes, per_block = request.block_hashes, self.units_per_block
        unit = self.hash_block_size
        points: dict[BlockHash, None] = {}
        units: dict[BlockHash, None] = {}
        for last in (prompt,):
            k = last // self.block_size
            if k and k * per_block <= len(hashes):
                points[hashes[k * per_block - 1]] = None
            # The states inside the last block too (its prompt tail): RAM
            # takes what no session owns yet for an orphan.
            for position in range(k * self.block_size + unit, last + 1, unit):
                if position // unit <= len(hashes):
                    units[hashes[position // unit - 1]] = None
        request_id = request.request_id
        if points:
            self._held_points[request_id] = tuple(points)
            self._kept.update(list(points))
        units.update(points)
        if units and self.ram_policy is not None:
            self._held_units[request_id] = tuple(units)
            self.ram_policy.hold(units)

    def _release_resume_points(self, session: BlockHash) -> None:
        kept = self._kept_by.pop(session, None)
        if kept is not None:
            self._release_points(kept[1])

    def _release_points(self, states: tuple[BlockHash, ...]) -> None:
        for state in states:
            self._kept[state] -= 1
            if self._kept[state] > 0:
                continue
            del self._kept[state]
            self._superseded[state] = None
            self._superseded.move_to_end(state)
            while len(self._superseded) > _MAX_SEEN_ENDS:
                self._superseded.popitem(last=False)
            self._drop_ram_state(state)
            if self.prune_gpu_states:
                self._prune_pending.append(state)

    def _drop_ram_state(self, state: BlockHash) -> None:
        """RAM drops the block-boundary state `state` no live session resumes
        at, unless it is spaced, a point several continuations share, or
        where a queued request resumes."""
        if self._shared(state) or state in self._queued_states:
            return
        cpu_pool = self.cpu_block_pool
        cpu = cpu_pool.cached_block_hash_to_block
        for group_id in self.mamba_gids:
            grouped = make_block_hash_with_group_id(state, group_id)
            for block in list(cpu.get_all_blocks(grouped)):
                num_tokens = block.block_hash_num_tokens
                if (
                    block.ref_cnt
                    or block.block_id in self._loading_cpu
                    or (num_tokens is not None and self._spaced(num_tokens))
                    # Another key (an exact-end state at a boundary).
                    or cpu_pool.cached_block_hashes_by_block.get(block.block_id)
                ):
                    continue
                if cpu_pool._maybe_evict_cached_block(block):
                    self.num_ram_states_dropped += 1

    def _prune_gpu_states(self, request: "Request") -> None:
        """Release the block-boundary states on a finished request's chain
        that RAM would not keep either (`_thinned`): its next turn resumes at
        its end, and each takes a block per Mamba group until the GPU's LRU
        order reaches it, which ranks them by age across sessions, not by use.
        Kept: spaced states, live sessions' resume points, branch points,
        states in use or where a waiting request resumes, and blocks that
        carry another key.

        The request still holds its blocks here (the scheduler frees them
        after `request_finished`), and a state in use cannot be released: its
        keys wait for the next steps (`_apply_prunes`)."""
        if not self.prune_gpu_states or self._checkpoints is None:
            return
        hashes, per_block = request.block_hashes, self.units_per_block
        for i in range(1, _end(request) // self.block_size + 1):
            if i * per_block > len(hashes):
                break
            if not self._spaced(i * self.block_size):
                self._prune_pending.append(hashes[i * per_block - 1])

    def _apply_prunes(self) -> None:
        """Release the pending states (`_prune_gpu_states`) nothing holds. A
        state a request holds stays pending for up to `_PRUNE_RETRY_STEPS`
        steps. `prune_outcomes` counts each state's outcome once.

        The checkpoint graph lets a superseded boundary state go while the
        running request still holds its blocks; they stay cached on the GPU
        with no unit. Such a state is evicted from the GPU cache directly
        ("pruned untracked"); a tracked one is released through the graph."""
        checkpoints = self._checkpoints
        pool = self._gpu_block_pool
        pending, self._prune_pending = self._prune_pending, []
        if not pending or checkpoints is None or pool is None:
            return
        graph = checkpoints.graph
        waiting = set(checkpoints._resume.values()) | self._queued_states
        gpu = pool.cached_block_hash_to_block
        outcomes = self.prune_outcomes
        retry: dict[BlockHash, None] = {}
        for key in dict.fromkeys(pending):
            unit = graph.units.get(key)
            outcome = (
                "kept"
                if key in self._kept
                else "frontier"
                if unit is not None and unit.frontier
                else "junction"
                if key in self._junctions
                else "branch point"
                if self._shared(key)
                else "held"
                if checkpoints.held(key)
                else "needed"
                if checkpoints._needed(key, waiting)
                else None
            )
            blocks: list[KVCacheBlock] = []
            if outcome is None:
                blocks = [
                    block
                    for group_id in self.mamba_gids
                    for block in gpu.get_all_blocks(
                        make_block_hash_with_group_id(key, group_id)
                    )
                ]
                if not blocks:
                    outcome = "no state"
                elif any(b.ref_cnt for b in blocks):
                    # Another request runs past it.
                    outcome = "in use"
                elif any(
                    pool.cached_block_hashes_by_block.get(b.block_id) for b in blocks
                ):
                    outcome = "other key"
            if outcome in ("held", "needed", "in use"):
                waited = self._prune_waits.get(key, 0) + 1
                if waited < _PRUNE_RETRY_STEPS:
                    self._prune_waits[key] = waited
                    retry[key] = None
                    continue
            self._prune_waits.pop(key, None)
            if outcome is None:
                if unit is not None:
                    checkpoints.release(key)
                else:
                    for block in blocks:
                        if pool._maybe_evict_cached_block(block):
                            checkpoints.frontiers.to_front(block)
                pruned = sum(b.block_hash is None for b in blocks)
                outcome = (
                    "not uncached"
                    if not pruned
                    else "pruned"
                    if unit is not None
                    else "pruned untracked"
                )
                self.num_gpu_states_pruned += pruned
            outcomes[outcome] += 1
        self._prune_pending.extend(retry)

    def _session_units(
        self, request: "Request", loop_start: int
    ) -> list[tuple[int, BlockHash]]:
        """The hash units a finished request's session owns: its block
        boundaries (pages and the states there), the hash units inside the
        last blocks of its prompt, of its end and of its loop's start (partial
        pages, the prompt tails' states), and its exact-end state."""
        unit, block_size = self.hash_block_size, self.block_size
        hashes = request.block_hashes
        end = _end(request)
        units = [
            (i * block_size, hashes[i * self.units_per_block - 1])
            for i in range(1, end // block_size + 1)
            if i * self.units_per_block <= len(hashes)
        ]
        for last in {loop_start, min(end, request.num_prompt_tokens), end}:
            start = last // block_size * block_size
            for position in range(start + unit, last + 1, unit):
                if position // unit <= len(hashes):
                    units.append((position, hashes[position // unit - 1]))
        checkpoints = self._checkpoints
        if checkpoints is not None and self._kv_cache_manager is not None:
            managers = self._kv_cache_manager.coordinator.single_type_managers
            entries = checkpoints.frontiers.entries
            for group_id in self.mamba_gids:
                for block in managers[group_id].req_to_blocks.get(
                    request.request_id, ()
                ):
                    if block.block_hash is None:
                        continue
                    entry = entries.get(get_block_hash(block.block_hash))
                    if entry is not None:
                        units.append((entry.num_tokens, entry.key))
        return units

    def reset(self) -> bool:
        for restore in list(self._restores.values()):
            if restore.done:
                self._release(restore)
            else:
                restore.reset = True
                del self._restores[restore.request_id]
        for restore in self._unsent:
            # Never sent: nothing reads its blocks.
            restore.reset = True
            self._loading_cpu.difference_update(c.block_id for _, c in restore.loads)
            self.cpu_block_pool.free_blocks([c for _, c in restore.loads])
            self._release(restore)
        self._unsent = []
        self.ram_frontiers.clear()
        self._kept.clear()
        self._kept_by.clear()
        self._held_points.clear()
        self._held_units.clear()
        self._superseded.clear()
        self._junctions.clear()
        self._queued_units = self._queued_states = set()
        if self._checkpoints is not None:
            self._checkpoints.queued_units = ()
        self._extends_end.clear()
        if self.ram_policy is not None:
            self.ram_policy.reset()
        self._waiting_since.clear()
        self._head = None
        self._unpin_head()
        self._queued.clear()
        self._prune_pending.clear()
        self._prune_waits.clear()
        if self._restores_by_event:
            super().reset()
            return False
        return super().reset()

    # ------------------------------------------------------------------
    # Measurement: tokens a resume recomputes that it had already seen.

    def _remember_end(self, request: "Request", loop_start: int) -> BlockHash | None:
        """Remember the end a finished request reached, its prompt's end and
        its loop's start: a reasoning model's next turn drops the reasoning
        after a user prompt, so it extends only the prompt that began the
        loop. Returns the end's key, its session's."""
        end = _end(request)
        key = self._remember(request, end)
        if key is None:
            return None
        for last in {min(end, request.num_prompt_tokens), loop_start}:
            if last < end:
                prompt_key = self._remember(request, last)
                if prompt_key is not None:
                    self._session_of[prompt_key] = key
        return key

    def _remember(self, request: "Request", num_tokens: int) -> BlockHash | None:
        frontiers = None if self._checkpoints is None else self._checkpoints.frontiers
        unit = self.hash_block_size
        if (
            frontiers is None
            or num_tokens <= 0
            or num_tokens // unit > len(request.block_hashes)
            or num_tokens > len(request.all_token_ids)
        ):
            return None
        key = frontier_key(frontiers.hash_fn, request, num_tokens, unit)
        if key in self._seen_ends.entries:
            self._seen_order.move_to_end(key)
            return key
        h = num_tokens // unit
        anchor = request.block_hashes[h - 1] if h else None
        start = max(h * unit, num_tokens - _SUFFIX)
        suffix = tuple(request.all_token_ids[start:num_tokens])
        self._seen_ends.add(_Unit(key, anchor, num_tokens, suffix, ()))
        self._seen_tails[key] = tuple(request.all_token_ids[h * unit : num_tokens])
        if h:
            # A turn that parts inside the end's own hash unit shares only
            # the unit before it.
            previous = request.block_hashes[h - 2] if h > 1 else None
            self._seen_by_previous.setdefault(previous, {})[key] = tuple(
                request.all_token_ids[(h - 1) * unit : num_tokens]
            )
            self._previous_of[key] = previous
        self._seen_order[key] = None
        while len(self._seen_order) > _MAX_SEEN_ENDS:
            oldest, _ = self._seen_order.popitem(last=False)
            self._seen_ends.remove(oldest)
            self._seen_tails.pop(oldest, None)
            self._session_of.pop(oldest, None)
            if oldest in self._previous_of:
                previous = self._previous_of.pop(oldest)
                by_previous = self._seen_by_previous[previous]
                del by_previous[oldest]
                if not by_previous:
                    del self._seen_by_previous[previous]
        return key

    def _seen_end(self, request: "Request") -> tuple[int, BlockHash | None, bool]:
        """How far `request` shares an end (or prompt end) a finished request
        reached, that end's session's key, and whether it extends the end
        itself: the end it extends, or else the deepest end whose hash unit
        it shares, up to where they part. A reasoning model's next turn parts
        from the previous prompt a few tokens before its end: that prompt
        ended with the generation prefix, which the next turn renders
        without."""
        frontiers = None if self._checkpoints is None else self._checkpoints.frontiers
        if frontiers is None or not self._seen_ends.entries:
            return 0, None, False
        unit = self.hash_block_size
        hashes = request.block_hashes
        tokens = request.all_token_ids
        for h in range(min(request.num_tokens // unit, len(hashes)), -1, -1):
            anchor = hashes[h - 1] if h else None
            lengths = self._seen_ends.index.get(anchor) or {}
            parted = self._seen_by_previous.get(anchor, {})
            if not lengths and not parted:
                continue
            for num_tokens in sorted(lengths, reverse=True):
                if num_tokens > request.num_tokens:
                    continue
                start = max(h * unit, num_tokens - _SUFFIX)
                keys = lengths[num_tokens].get(tuple(tokens[start:num_tokens]))
                if not keys:
                    continue
                key = frontier_key(frontiers.hash_fn, request, num_tokens, unit)
                if key in keys:
                    session = self._session_of.get(key)
                    return num_tokens, session or key, session is None
            # No end here that it extends: the one it shares most of, its own
            # hash unit's or (parted inside it) the next one's.
            best, best_key = -1, None
            ours = tokens[h * unit : h * unit + unit]
            tails = [
                (key, self._seen_tails.get(key, ()))
                for by_suffix in lengths.values()
                for keys in by_suffix.values()
                for key in keys
            ]
            tails.extend(parted.items())
            for key, tail in tails:
                shared = 0
                for mine, theirs in zip(ours, tail):
                    if mine != theirs:
                        break
                    shared += 1
                if shared > best:
                    best, best_key = shared, key
            if best_key is not None:
                return (
                    h * unit + best,
                    self._session_of.get(best_key, best_key),
                    False,
                )
        return 0, None, False

    def observe_admission(self, request: "Request", num_computed_tokens: int) -> None:
        """Record how many tokens the admitted request recomputes that a
        finished request had already processed, by the tier it resumed from."""
        request_id = request.request_id
        if request_id in self._observed:
            return
        self._observed.add(request_id)
        restored = request_id in self._admitted_from_ram
        self._admitted_from_ram.discard(request_id)
        seen, key, extends_end = self._seen_end(request)
        self._hold_prefill_points(request, key, extends_end)
        self._note_junction(request, num_computed_tokens, seen)
        if key is None:
            return
        # The session resumed, whichever tier holds it.
        self._resumed_from[request_id] = key
        if extends_end:
            self._extends_end.add(request_id)
        if self.ram_policy is not None:
            self.ram_policy.resumed(key)
        tier = "ram" if restored else "gpu" if num_computed_tokens else "cold"
        self.resume_stats.record(tier, max(0, seen - num_computed_tokens))

    def _note_junction(
        self, request: "Request", num_computed_tokens: int, seen: int
    ) -> None:
        """Remember a shared prefix's junction: where the admitted `request`'s
        prefill stops to cache a state (`Request.shared_prefix_boundary`, to
        its block), when that is past the point it resumes at and past the end
        of a finished request it continues (`seen`). Another request wrote the
        pages under it: a system prompt a new session or a compaction starts
        with, an earlier message edited twice. Both tiers keep its state as a
        point several continuations share (`_shared`), so the next request
        with the same prefix resumes at it. A block-boundary state the request
        resumes at past such an end is a junction too: another conversation
        left it (sessions that started together computed their system prompt
        at once, a branch point the checkpoint graph forgets once it rebuilds
        the unit from RAM). A resume at a junction keeps it remembered. A
        session that re-reads its own history, its states lost, marks none:
        its next turn resumes at its new end."""
        hashes, unit = request.block_hashes, self.hash_block_size
        if num_computed_tokens and not num_computed_tokens % self.block_size:
            resumed = hashes[num_computed_tokens // unit - 1]
            if resumed in self._junctions:
                self._junctions.move_to_end(resumed)
            elif num_computed_tokens > seen:
                self._add_junction(request, resumed)
        end = request.shared_prefix_boundary // self.block_size * self.block_size
        if end <= max(num_computed_tokens, seen) or end // unit > len(hashes):
            return
        self._add_junction(request, hashes[end // unit - 1])

    def _add_junction(self, request: "Request", key: BlockHash) -> None:
        junctions = self._junctions
        junctions[key] = None
        junctions.move_to_end(key)
        while len(junctions) > _MAX_SEEN_ENDS:
            junctions.popitem(last=False)
        if self.ram_policy is not None:
            # Until its session owns the state, RAM would take it for an
            # orphan (`_hold_prefill_points`).
            request_id = request.request_id
            held = self._held_units.get(request_id, ())
            if key not in held:
                self._held_units[request_id] = (*held, key)
                self.ram_policy.hold((key,))

    # ------------------------------------------------------------------
    # Stats

    def get_stats(self) -> SimpleCPUOffloadStats:
        stats = super().get_stats()
        current = self.restore_stats
        for event, field_name in RESTORE_EVENT_TO_FIELD.items():
            delta = getattr(current, field_name) - getattr(
                self._restore_snapshot, field_name
            )
            if delta > 0:
                stats.increase_counter(TieredMetricName.RESTORES, delta, (event,))
        self._restore_snapshot = replace(current)
        resumes, previous = self.resume_stats, self._resume_snapshot
        for tier, count in resumes.counts.items():
            delta = count - previous.counts.get(tier, 0)
            if delta > 0:
                stats.increase_counter(TieredMetricName.RESUMES, delta, (tier,))
        for tier, tokens in resumes.tokens.items():
            delta = tokens - previous.tokens.get(tier, 0)
            if delta > 0:
                stats.increase_counter(
                    TieredMetricName.RESUME_RECOMPUTE_TOKENS, delta, (tier,)
                )
        for labels, count in resumes.buckets.items():
            delta = count - previous.buckets.get(labels, 0)
            if delta > 0:
                stats.increase_counter(
                    TieredMetricName.RESUMES_BY_RECOMPUTE, delta, labels
                )
        self._resume_snapshot = _ResumeStats(
            dict(resumes.counts), dict(resumes.tokens), dict(resumes.buckets)
        )
        stats.set_gauge(TieredMetricName.RAM_FRONTIERS, len(self.ram_frontiers.entries))
        pruned = self.num_gpu_states_pruned - self._gpu_states_pruned_snapshot
        if pruned > 0:
            stats.increase_counter(TieredMetricName.GPU_STATES_PRUNED, pruned, ())
        self._gpu_states_pruned_snapshot = self.num_gpu_states_pruned
        for outcome, count in self.prune_outcomes.items():
            delta = count - self._prune_outcomes_snapshot[outcome]
            if delta > 0:
                stats.increase_counter(
                    TieredMetricName.GPU_STATE_PRUNES, delta, (outcome,)
                )
        self._prune_outcomes_snapshot = Counter(self.prune_outcomes)
        dropped = self.num_ram_states_dropped - self._ram_states_dropped_snapshot
        if dropped > 0:
            stats.increase_counter(TieredMetricName.RAM_STATES_DROPPED, dropped, ())
        self._ram_states_dropped_snapshot = self.num_ram_states_dropped
        pool = self._gpu_block_pool
        if pool is not None:
            stats.set_gauge(
                TieredMetricName.GPU_STATE_BLOCKS,
                sum(
                    block.block_hash is not None
                    and get_group_id(block.block_hash) in self._mamba_gid_set
                    for block in pool.blocks
                ),
            )
        if self.ram_policy is not None:
            current, previous = self.ram_policy.stats, self._ram_evictions_snapshot
            for kind, field_name in (
                ("orphan", "orphans"),
                ("session", "session_blocks"),
                ("dead", "dead"),
                ("duplicate", "duplicates"),
                ("queued", "queued"),
                ("unranked", "unranked"),
            ):
                delta = getattr(current, field_name) - getattr(previous, field_name)
                if delta > 0:
                    stats.increase_counter(
                        TieredMetricName.RAM_EVICTIONS, delta, (kind,)
                    )
            self._ram_evictions_snapshot = replace(current, counts={})
        return stats
