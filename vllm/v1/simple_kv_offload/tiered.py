# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler side of TieredCPUOffloadConnector: a RAM tier that extends VRAM.

VRAM -> RAM: the GPU prefix cache evicts least recently used first; the lazy
walk of `SimpleCPUOffloadScheduler` copies every block a lookup can use before
the GPU may drop it, including the blocks of exact-end and prompt-tail states
(keys that end inside a block), each with the other groups' blocks under the
same keys.

RAM -> VRAM ("promote, then hit"): before a waiting request's next lookup,
the connector finds the deepest point the request can resume at with blocks
from either tier, loads the blocks only RAM holds into GPU blocks, caches them
under their own keys (re-indexing an exact-end state as a GPU frontier), and
pins them until the request is admitted. The ordinary GPU lookup then makes
the hit; no tokens are ever loaded as an external suffix, so a session split
across the tiers in any order resumes whole.
"""

import bisect
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
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
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.metrics import SimpleCPUOffloadStats, TieredMetricName

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
_PIN_TIMEOUT_S = 30.0
# A request waits at most this long for room to restore into, then resumes
# from the GPU alone.
_MAX_RESTORE_WAIT_S = 10.0
# Finished requests' ends remembered to measure resumes against.
_MAX_SEEN_ENDS = 65536
# Upper bounds of the recompute buckets (tokens recomputed that a resume had
# already seen).
RECOMPUTE_BUCKETS = (0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536)


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
    # Requests that resumed from the GPU alone after waiting too long.
    gave_up: int = 0
    # Restores whose request was not admitted before the pins timed out.
    pin_timeouts: int = 0
    # Restores whose request left before the load finished.
    orphaned: int = 0


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
        **kwargs: Any,
    ) -> None:
        kwargs["lazy_offload"] = True
        super().__init__(*args, **kwargs)
        if ram_eviction not in RAM_EVICTION_POLICIES:
            raise ValueError(
                f"ram_eviction must be one of {RAM_EVICTION_POLICIES}, "
                f"got {ram_eviction!r}"
            )
        if ram_eviction == "gdsf":
            raise NotImplementedError("ram_eviction='gdsf' is not implemented yet")
        self.ram_eviction = ram_eviction
        self.prefill_cost_coefficients = prefill_cost_coefficients
        self.frequency_half_life_s = frequency_half_life_s
        groups = self.cpu_kv_cache_config.kv_cache_groups
        self.mamba_gids = [
            g
            for g in self.prefix_cacheable_group_ids
            if isinstance(groups[g].kv_cache_spec, MambaSpec)
        ]
        self.units_per_block = self.block_size // self.hash_block_size
        self._kv_cache_manager: KVCacheManager | None = None
        # Exact-end states in RAM, and the ends finished requests reached.
        self.ram_frontiers = _EndIndex()
        self._seen_ends = _EndIndex()
        self._seen_order: OrderedDict[BlockHash, None] = OrderedDict()
        # Restores: by request, queued for the next load event, by event.
        self._restores: dict[str, _Restore] = {}
        self._unsent: list[_Restore] = []
        self._restores_by_event: dict[int, list[_Restore]] = {}
        self._load_event_pending_counts: dict[int, int] = {}
        self._loading_cpu: set[int] = set()
        self._num_pinned = 0
        # Request -> when it first waited for room to restore into.
        self._waiting_since: dict[str, float] = {}
        self._observed: set[str] = set()
        # Admitted at a point a restore loaded, until `observe_admission`.
        self._admitted_from_ram: set[str] = set()
        self.restore_stats = RestoreStats()
        self._restore_snapshot = RestoreStats()
        self.resume_stats = _ResumeStats()
        self._resume_snapshot = _ResumeStats()
        self.cpu_block_pool.key_removal_listener = self._ram_keys_removed

    # ------------------------------------------------------------------
    # Binding

    def bind_kv_cache_manager(self, kv_cache_manager: "KVCacheManager") -> None:
        self._kv_cache_manager = kv_cache_manager

    @property
    def _checkpoints(self) -> "HybridCheckpointManager | None":
        manager = self._kv_cache_manager
        return None if manager is None else manager.checkpoints

    @property
    def _max_pinned(self) -> int:
        assert self._gpu_block_pool is not None
        return max(1, self._gpu_block_pool.num_gpu_blocks // 4)

    # ------------------------------------------------------------------
    # VRAM -> RAM

    def _restorable(self, block: KVCacheBlock) -> bool:
        """Every key a lookup can use is restored, including keys that end
        inside a block (exact-end and prompt-tail states)."""
        return block.block_hash is not None

    def needs_spill(self, block: KVCacheBlock) -> bool:
        """SpillGuard: also a block of an exact-end state whose other blocks
        RAM does not hold yet. The GPU drops such a state whole, so a block
        copied ahead of the others leaving first would take them along
        uncopied."""
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
        for key in (
            block.block_hash,
            *pool.cached_block_hashes_by_block.get(block.block_id, ()),
        ):
            entry = frontiers.entries.get(get_block_hash(key))
            if entry is None:
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
        request is pending (the scheduler retries it next step), else
        (0, False)."""
        request_id = request.request_id
        restore = self._restores.get(request_id)
        if restore is not None:
            return (0, False) if restore.done else (None, False)
        if request.skip_reading_external_kv or self._gpu_block_pool is None:
            return 0, False
        plan = self._plan(request)
        if plan is None:
            self._waiting_since.pop(request_id, None)
            return 0, False
        return self._start_restore(request, plan)

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
        if (
            self._num_pinned
            and self._num_pinned + num_blocks + len(pinned) > self._max_pinned
        ):
            # Restores that wait for admission already pin their share; a
            # restore alone may pin a whole session.
            stats.waited_cap += 1
            return self._wait(request_id)
        # Blocks pinned below leave the free queue: they are not safe to take.
        if pool.num_spill_safe_free_blocks(num_blocks, exclude=pinned) < num_blocks:
            stats.waited_guard += 1
            self.on_allocation_held(num_blocks)
            return self._wait(request_id)
        self._waiting_since.pop(request_id, None)
        pool.touch(pinned)
        gpu_blocks = pool.get_new_blocks(num_blocks)
        self.cpu_block_pool.touch(loads)
        restore = _Restore(
            request_id,
            plan.num_tokens,
            list(zip(gpu_blocks, loads)),
            pinned,
            plan.frontier,
            plan.state,
        )
        self._restores[request_id] = restore
        self._unsent.append(restore)
        self._loading_cpu.update(b.block_id for b in loads)
        self._num_pinned += restore.num_pinned
        stats.started += 1
        return None, False

    def _wait(self, request_id: str) -> tuple[int | None, bool]:
        """Wait a step for room to restore into, at most _MAX_RESTORE_WAIT_S."""
        now = time.monotonic()
        since = self._waiting_since.setdefault(request_id, now)
        if now - since < _MAX_RESTORE_WAIT_S:
            return None, False
        self._waiting_since.pop(request_id, None)
        self.restore_stats.gave_up += 1
        return 0, False

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
        return deepest

    def _resume_points(self, request: "Request") -> Iterator[_Plan]:
        """Points `request` can resume at with blocks from either tier,
        deepest first: exact-end states (GPU frontiers and RAM ones), then
        states at hash-unit boundaries."""
        pool = self._gpu_block_pool
        assert pool is not None
        gpu = pool.cached_block_hash_to_block
        cpu = self.cpu_block_pool.cached_block_hash_to_block

        def find(key: BlockHashWithGroupId) -> _Piece | None:
            block = gpu.get_one_block(key)
            if block is not None:
                return _Piece(block, True)
            block = cpu.get_one_block(key)
            # A block another restore is loading counts: the restore waits
            # for that load (`_start_restore`).
            return None if block is None else _Piece(block, False)

        unit, per_block = self.hash_block_size, self.units_per_block
        hashes = request.block_hashes
        max_length = request.num_tokens - 1
        # Full-attention pages either tier holds, from the first.
        chain: list[_Piece] = []
        for i in range(min(len(hashes) // per_block, max_length // self.block_size)):
            piece = find(
                make_block_hash_with_group_id(
                    hashes[(i + 1) * per_block - 1], self.fa_gidx
                )
            )
            if piece is None:
                break
            chain.append(piece)
        checkpoints = self._checkpoints
        frontiers = None if checkpoints is None else checkpoints.frontiers
        hash_fn = None if frontiers is None else frontiers.hash_fn
        tokens = request.all_token_ids
        for h in range(min(max_length // unit, len(hashes)), -1, -1):
            anchor = hashes[h - 1] if h else None
            if hash_fn is not None:
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
            if k > len(chain):
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
                yield _Plan(num_tokens, pieces, state=(block_hash, num_tokens))

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
            if restore.done and now - restore.done_at > _PIN_TIMEOUT_S:
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
        self._waiting_since.pop(request_id, None)
        self._observed.discard(request_id)
        self._admitted_from_ram.discard(request_id)
        self._remember_end(request)
        return super().request_finished(request, block_ids)

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
        self._waiting_since.clear()
        if self._restores_by_event:
            super().reset()
            return False
        return super().reset()

    # ------------------------------------------------------------------
    # Measurement: tokens a resume recomputes that it had already seen.

    def _remember_end(self, request: "Request") -> None:
        frontiers = None if self._checkpoints is None else self._checkpoints.frontiers
        num_tokens = request.num_computed_tokens
        unit = self.hash_block_size
        if (
            frontiers is None
            or num_tokens <= 0
            or num_tokens // unit > len(request.block_hashes)
            or num_tokens > len(request.all_token_ids)
        ):
            return
        key = frontier_key(frontiers.hash_fn, request, num_tokens, unit)
        if key in self._seen_ends.entries:
            self._seen_order.move_to_end(key)
            return
        h = num_tokens // unit
        anchor = request.block_hashes[h - 1] if h else None
        start = max(h * unit, num_tokens - _SUFFIX)
        suffix = tuple(request.all_token_ids[start:num_tokens])
        self._seen_ends.add(_Unit(key, anchor, num_tokens, suffix, ()))
        self._seen_order[key] = None
        while len(self._seen_order) > _MAX_SEEN_ENDS:
            oldest, _ = self._seen_order.popitem(last=False)
            self._seen_ends.remove(oldest)

    def _seen_end(self, request: "Request") -> int:
        """The longest end a finished request reached that `request` extends."""
        frontiers = None if self._checkpoints is None else self._checkpoints.frontiers
        if frontiers is None or not self._seen_ends.entries:
            return 0
        unit = self.hash_block_size
        hashes = request.block_hashes
        tokens = request.all_token_ids
        for h in range(min(request.num_tokens // unit, len(hashes)), -1, -1):
            lengths = self._seen_ends.index.get(hashes[h - 1] if h else None)
            if not lengths:
                continue
            for num_tokens in sorted(lengths, reverse=True):
                if num_tokens > request.num_tokens:
                    continue
                start = max(h * unit, num_tokens - _SUFFIX)
                keys = lengths[num_tokens].get(tuple(tokens[start:num_tokens]))
                if keys and (
                    frontier_key(frontiers.hash_fn, request, num_tokens, unit) in keys
                ):
                    return num_tokens
        return 0

    def observe_admission(self, request: "Request", num_computed_tokens: int) -> None:
        """Record how many tokens the admitted request recomputes that a
        finished request had already processed, by the tier it resumed from."""
        request_id = request.request_id
        if request_id in self._observed:
            return
        self._observed.add(request_id)
        restored = request_id in self._admitted_from_ram
        self._admitted_from_ram.discard(request_id)
        seen = self._seen_end(request)
        if not seen:
            return
        tier = "ram" if restored else "gpu" if num_computed_tokens else "cold"
        self.resume_stats.record(tier, max(0, seen - num_computed_tokens))

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
        return stats
