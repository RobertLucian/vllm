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
import logging
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Collection, Iterable, Iterator
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
from vllm.v1.request import RequestStatus
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.metrics import SimpleCPUOffloadStats, TieredMetricName

from vllm.v1.simple_kv_offload.ram_policy import (
    LADDER_RANKINGS,
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
ADMISSION_MODES = ("pack", "pass", "legacy")

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
# A chunked restore that loads nothing for this long gives up the rest
# (`_advance_chunked`): its loaded chunks stay cached on the GPU.
_CHUNK_STALL_S = 10.0
# Inline spills: the chunks a chunked restore may have taken whose loads have
# not landed. Nothing else paces it: its chunks wait for no copies, and each
# spills as many GPU blocks into RAM in its step.
_INLINE_CHUNKS_UNACKED = 2
# A lost session's eviction kinds, the most telling first (`_report_losses`).
_LOSS_KIND_ORDER = (
    "gpu",
    "queued",
    "leaving",
    "unranked",
    "session",
    "rung",
    "duplicate",
    "orphan",
)
# What each tier holds, by kind of block (`_tier_contents`), recomputed this
# often; RAM blocks no eviction ranks (`RamGreedyDualSizeFrequency.
# unranked_blocks`), this often.
TIER_BLOCK_KINDS = (
    "free",
    "in_use",
    "orphan",
    "page",
    "page_exact_end",
    "exact_end",
    "queued",
    "held",
    "near_end",
    "ladder",
    "shared",
    "prompt_tail",
    "other",
)
_TIER_BLOCKS_EVERY_S = 5.0
_UNRANKED_EVERY_S = 30.0
# Ladder points by why they went or what they did (`_unkeep_point`,
# `_note_ladder_outcome`): dropped (no cut could stop there, or by value),
# RAM blocks the ranked drops freed, and of those the copies stored since
# the last two evictions (copied only to be dropped); resumes at a ladder
# point and the tokens it saved over the point below; resumes below a
# dropped one, by why it went, and the tokens between.
LADDER_EVENTS = (
    "dropped_safe",
    "dropped_ranked",
    "dropped_ranked_blocks",
    "dropped_fresh_blocks",
    "payoff_resumes",
    "payoff_tokens",
    "regret_safe_resumes",
    "regret_safe_tokens",
    "regret_ranked_resumes",
    "regret_ranked_tokens",
)
# Sessions whose dropped ladder points are remembered (`_dropped_rungs`).
_MAX_DROPPED_SESSIONS = 4096
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
    # Restores that loaded in chunks (`_start_chunked`), their chunks, and
    # those that gave up the rest (a stall, or the spill guard's).
    chunked: int = 0
    chunks: int = 0
    chunked_aborted: int = 0
    # Inline spills: whole restores and chunks that went although their
    # blocks' copies cost more than a chunk or than RAM had room for, nothing
    # they could wait for being under way (no load, no chunked restore).
    started_without_room: int = 0
    # The fields below are exported as `ADMISSION_EVENT_TO_FIELD` events,
    # `SEARCH_EVENT_TO_FIELD` events, `SCHEDULER_PART_TO_FIELD` seconds and
    # `PEAK_KIND_TO_FIELD` gauges.
    # Steps a restore waited for room, split by whether it asked the walk to
    # copy GPU blocks out for it; and restores that waited that long, from
    # their first wait to their start or giving up (`_stop_waiting`).
    head_waits_copying: int = 0
    head_waits_not_copying: int = 0
    head_waits_over_1s: int = 0
    head_waits_over_10s: int = 0
    head_waits_over_60s: int = 0
    # Requests with nothing to restore that `admission="pack"` let past a
    # waiting restore (legacy admission held them), and of those, the ones
    # with no resume point at all.
    passed_head: int = 0
    passed_head_cold: int = 0
    # Pack admission. Restores admitted in the first scheduling pass after
    # their load landed, the next, within four, later.
    admit_steps_0: int = 0
    admit_steps_1: int = 0
    admit_steps_2_4: int = 0
    admit_steps_over_4: int = 0
    # The blocks a restore held for its tokens past the resume point
    # (`_estimate_extra`): admissions that took exactly as many, that needed
    # more (and how many more), blocks they did not need, blocks released
    # with no admission; re-estimates as the load landed that grew or shrank
    # the room, and by how many blocks; estimates whose first chunk the
    # scheduler would not schedule (0 tokens).
    extra_exact_admissions: int = 0
    extra_short: int = 0
    extra_short_blocks: int = 0
    extra_unused_blocks: int = 0
    extra_released_unadmitted: int = 0
    extra_regrown: int = 0
    extra_regrown_blocks: int = 0
    extra_shrunk: int = 0
    extra_shrunk_blocks: int = 0
    extra_zero_chunk: int = 0
    # Restored requests admitted at another point than the restore's: short
    # of it (the GPU lost a piece meanwhile), past it.
    admitted_hit_short: int = 0
    admitted_hit_past: int = 0
    # Restores started in one step: the most, and steps with two or more;
    # the most restores loading at once.
    starts_per_step_max: int = 0
    multi_start_steps: int = 0
    concurrent_restores_max: int = 0
    # The most chunks a chunked restore had taken whose loads had not landed.
    chunks_unacked_max: int = 0
    # Steps with a claimant (`_update_claim`), allocations and restores that
    # fit but would have taken its room, and changes of claimant.
    claim_wait_steps: int = 0
    # Steps whose earliest request that did not fit had not yet waited
    # `claim_after_s`: no one left it room.
    claim_deferred_steps: int = 0
    claim_blocked_admissions: int = 0
    claim_blocked_restores: int = 0
    claimant_changes: int = 0
    # Blocks the walk copied out one chunk ahead of a chunked restore.
    copy_ahead_blocks: int = 0
    # Searches for a resume point (`_search`): calls and points visited; with
    # `verify_search`, the whole walk's points too, and lookups whose bounded
    # search found another decision or point than the whole walk (must stay
    # 0), or the same with another GPU-only point (the pin timeout's).
    search_calls: int = 0
    search_points: int = 0
    search_points_whole: int = 0
    # Hash units the searches scanned (and, verifying, the whole walks).
    search_units: int = 0
    search_units_whole: int = 0
    search_mismatches: int = 0
    search_gpu_only_differs: int = 0
    # Scheduler time the connector took (ns): its lookups for waiting
    # requests (of which: the search, the junction's chain, the room checks
    # of a restore), its queue observation, its per-step work, its stats.
    lookup_ns: int = 0
    lookup_search_ns: int = 0
    lookup_junction_ns: int = 0
    lookup_room_ns: int = 0
    # The room estimates' share of that (`_estimate_extra`: the junction, the
    # scheduler's sizing and block count), at landing too.
    estimate_ns: int = 0
    observe_queue_ns: int = 0
    step_ns: int = 0
    stats_ns: int = 0


# event label value -> RestoreStats field name: the fields above the
# unexported ones.
RESTORE_EVENT_TO_FIELD = {
    f: f
    for f in (
        "started",
        "completed",
        "restored_blocks",
        "restored_frontiers",
        "waited_guard",
        "waited_cap",
        "waited_behind",
        "gave_up",
        "pin_timeouts",
        "orphaned",
        "unpinned_for_spill",
        "chunked",
        "chunks",
        "chunked_aborted",
        "started_without_room",
    )
}
# `vllm:tiered_kv_admission_total{event}`: how waiting requests got room.
ADMISSION_EVENT_TO_FIELD = {
    f: f
    for f in (
        "head_waits_copying",
        "head_waits_not_copying",
        "head_waits_over_1s",
        "head_waits_over_10s",
        "head_waits_over_60s",
        "passed_head",
        "passed_head_cold",
        "admit_steps_0",
        "admit_steps_1",
        "admit_steps_2_4",
        "admit_steps_over_4",
        "extra_exact_admissions",
        "extra_short",
        "extra_short_blocks",
        "extra_unused_blocks",
        "extra_released_unadmitted",
        "extra_regrown",
        "extra_regrown_blocks",
        "extra_shrunk",
        "extra_shrunk_blocks",
        "extra_zero_chunk",
        "admitted_hit_short",
        "admitted_hit_past",
        "multi_start_steps",
        "claim_wait_steps",
        "claim_deferred_steps",
        "claim_blocked_admissions",
        "claim_blocked_restores",
        "claimant_changes",
        "copy_ahead_blocks",
    )
}
# `vllm:tiered_kv_search_total{event}`.
SEARCH_EVENT_TO_FIELD = {
    "calls": "search_calls",
    "points": "search_points",
    "points_whole": "search_points_whole",
    "units": "search_units",
    "units_whole": "search_units_whole",
    "mismatches": "search_mismatches",
    "gpu_only_differs": "search_gpu_only_differs",
}
# `vllm:tiered_kv_scheduler_seconds_total{part}`, from ns.
SCHEDULER_PART_TO_FIELD = {
    "lookup": "lookup_ns",
    "lookup_search": "lookup_search_ns",
    "lookup_junction": "lookup_junction_ns",
    "lookup_room": "lookup_room_ns",
    "estimate": "estimate_ns",
    "observe_queue": "observe_queue_ns",
    "step": "step_ns",
    "stats": "stats_ns",
}
# `vllm:tiered_kv_restore_peaks{kind}` gauges: maxima since start.
PEAK_KIND_TO_FIELD = {
    "starts_per_step": "starts_per_step_max",
    "concurrent_restores": "concurrent_restores_max",
    "chunks_unacked": "chunks_unacked_max",
}
# Why a waiting request did not go this step (`_last_wait`), and why a restore
# that had landed was not admitted (`landed_waits`): its restore loads, waits
# behind another (legacy), waits for room beside other restores' pins and
# room or for the claimant's, waits for copies, or another restore loads its
# RAM blocks; its allocation found no room, would take room others keep
# ("reserved") or the claimant's ("claim"), waits for copies, did not fit
# whole, or needs a state cached this step; the scheduler scheduled nothing
# for it (prefill deferred this step, a padded speculative step that does not
# fit, the low-priority cap, a chunk of 0 tokens after alignment or
# trimming; "scheduler" for any other cause) or did not reach it
# ("not_reached").
WAIT_REASONS = (
    "loading",
    "behind",
    "room_restores",
    "room_claim",
    "copies_restore",
    "same_blocks",
    "no_room",
    "reserved",
    "claim",
    "copies",
    "full_sequence",
    "same_step_state",
    "deferred_step",
    "spec_padding",
    "low_priority_cap",
    "aligned_chunk_zero",
    "trimmed_chunk_zero",
    "scheduler",
    "not_reached",
)
# What an admitted request resumed its session at (`_point_kind`).
RESUME_POINT_KINDS = (
    "exact_end",
    "near_end",
    "prompt_tail",
    "ladder",
    "shared",
    "other",
    "none",
)
# Admissions remembered, the most recent (`recent_resumes`).
_RECENT_RESUMES = 4096


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
    # Pieces before this index are the attention pages; the point's own
    # (states, an exact end, the page they end in, windows) follow.
    num_chain: int = 0

    @property
    def on_gpu_only(self) -> bool:
        return all(p.on_gpu for p in self.pieces)


@dataclass(slots=True)
class _Bound:
    """A search's line (`_search`): points of `floor` tokens or fewer cannot
    change a bounded search's decision, so `_resume_points` stops before the
    hash units that hold only such points (`stopped`). The units it scanned."""

    floor: int = -1
    stopped: bool = False
    units: int = 0


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
    # A chunked restore (`_start_chunked`): RAM blocks still to load, pinned,
    # the point's own last (`num_tail` of them); chunks sent and not landed;
    # its place in the queue; when it last took GPU blocks; whether it gave
    # up the rest.
    chunked: bool = False
    pending: list[KVCacheBlock] = field(default_factory=list)
    num_tail: int = 0
    in_flight: int = 0
    position: tuple[int, float, str] = (0, 0.0, "")
    progress_at: float = 0.0
    aborted: bool = False
    # Pack admission: spill-safe free GPU blocks the others leave for the
    # request's tokens past the point (`_estimate_extra`, a count: the blocks
    # stay cached, and in the free queue, until its allocation takes them),
    # and the first chunk's tokens they are for; the scheduling step its load
    # landed in; the request, for the re-estimate as it lands; when it began
    # waiting for room (logs).
    extra: int = 0
    chunk_tokens: int = 0
    done_step: int = 0
    request: "Request | None" = None
    started_at: float = 0.0
    # At admission, the room it kept and the blocks its request's allocation
    # took (-1: not yet).
    kept: int = 0
    used: int = -1

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
        restore_chunk_blocks: int = 16,
        ram_ladder_blocks: int = 8,
        ram_ladder_ranking: str = "fixed",
        abandon_gap_multiplier: float = 2.5,
        abandon_floor_s: float = 30.0,
        admission: str = "legacy",
        claim_after_s: float = 1.0,
        bounded_search: bool = True,
        verify_search: bool = False,
        log_restores: bool = False,
        summary_log_s: float = 60.0,
        **kwargs: Any,
    ) -> None:
        kwargs["lazy_offload"] = True
        super().__init__(*args, **kwargs)
        if restore_chunk_blocks < 0:
            raise ValueError("restore_chunk_blocks must be 0 (off) or more")
        if ram_ladder_blocks < 0:
            raise ValueError("ram_ladder_blocks must be 0 (off) or more")
        if ram_ladder_ranking not in LADDER_RANKINGS:
            raise ValueError(
                f"ram_ladder_ranking must be one of {LADDER_RANKINGS}, "
                f"got {ram_ladder_ranking!r}"
            )
        if admission not in ADMISSION_MODES:
            raise ValueError(
                f"admission must be one of {ADMISSION_MODES}, got {admission!r}"
            )
        if claim_after_s < 0:
            raise ValueError("claim_after_s must be 0 or more")
        # How waiting requests get room (module docstring). "pack": every
        # request that fits goes, restores side by side, each with room for
        # its tokens past the point; the earliest that does not fit claims
        # the room it waits for once it has waited `claim_after_s`
        # (`_update_claim`). "legacy": the requests behind a restore waiting
        # for room wait with it (`_behind_head`). "pass": legacy, but those
        # with nothing to restore pass it.
        self._pack = admission == "pack"
        self._pass = admission in ("pack", "pass")
        # Claiming at once kept restores that fit out whenever a session of
        # similar size waited (window R, 2026-11-24: 292 turns against 359
        # for "pass" in window Q at TP 1, 288 against legacy's 343 at TP 2);
        # most of those got room within a step or two anyway.
        self._claim_after_s = claim_after_s
        # Pack: the claimant (request id, queue position, blocks it needs),
        # the restores waiting for room and the blocks each needs, the
        # waiting requests whose allocation the spill guard held this step
        # (position, blocks), the scheduling step, and restores started in it.
        self._claim: tuple[str, tuple[int, float, str], int] | None = None
        # Pack: the room restores keep for their requests' tokens past the
        # point (the sum of `_Restore.extra`).
        self._reserved = 0
        self._waiting_need: dict[str, int] = {}
        self._held_now: dict[str, tuple[tuple[int, float, str], int]] = {}
        # Since when each queued request's allocation has been held.
        self._alloc_held_since: dict[str, float] = {}
        self._step = 0
        self._starts_this_step = 0
        # The scheduler's sizing of a restored request's first chunk past the
        # point, its tokens and blocks (`Scheduler.first_chunk`): the room a
        # restore keeps for it (`_estimate_extra`).
        self._first_chunk: (
            Callable[[Request, int, int, tuple[int, ...]], tuple[int, int]] | None
        ) = None
        # A lookup searches down from the deepest point only as far as the
        # restore gain it checks (`_search`); `verify_search` checks each such
        # search against the whole walk.
        self._bounded_search = bounded_search
        self._verify_search = verify_search
        # Why each waiting request did not go this step (`WAIT_REASONS`), and
        # why landed restores were not admitted, by reason; lookups by outcome.
        # The landed restores eligible in this step's scheduling, those it
        # looked up, and those `allocate_slots` refused.
        self._last_wait: dict[str, str] = {}
        self.landed_waits: Counter[str] = Counter()
        self._landed_waits_snapshot: Counter[str] = Counter()
        self.lookup_outcomes: Counter[str] = Counter()
        self._lookup_outcomes_snapshot: Counter[str] = Counter()
        self._landed_eligible: set[str] = set()
        self._landed_looked_up: set[str] = set()
        self._landed_refused: set[str] = set()
        # The point each restored request's restore loaded, from its admission
        # to `observe_admission`; the most recent admissions' tier and point.
        self._admitted_points: dict[str, int] = {}
        self.recent_resumes: OrderedDict[str, tuple[str, int]] = OrderedDict()
        # Logs: each restore's start, landing and admission at this level; a
        # summary line at most every `summary_log_s` after activity.
        self._restore_log_level = logging.INFO if log_restores else logging.DEBUG
        self._summary_log_s = summary_log_s
        self._summary_at = time.monotonic()
        self._summary_step = 0
        self._summary_snapshot = RestoreStats()
        self._summary_outcomes: Counter[str] = Counter()
        self._summary_landed: Counter[str] = Counter()
        self._summary_spill = (0, 0, 0, 0)
        self._summary_held_allocations = 0
        self._spill_ns_snapshot = (0, 0)
        # A session's checkpoint ladder (`_ladder`): its first rung this many
        # blocks back from its end, each further one twice as far. Its rungs
        # by session (state -> position) while their states may go on their
        # own (`ram_ladder_ranking`, RAM GDSF only); the positions of those
        # that went and why, carried to the session's later turns; ladder
        # events (`LADDER_EVENTS`).
        self._ladder_blocks = ram_ladder_blocks
        self._rungs_by: dict[BlockHash, dict[BlockHash, int]] = {}
        self._dropped_rungs: OrderedDict[BlockHash, list[tuple[int, str]]] = (
            OrderedDict()
        )
        self.ladder_events: Counter[str] = Counter()
        self._ladder_events_snapshot: Counter[str] = Counter()
        # Resumes by the kind of point they landed on and the tier
        # (`RESUME_POINT_KINDS`).
        self.resume_points: Counter[tuple[str, str]] = Counter()
        self._resume_points_snapshot: Counter[tuple[str, str]] = Counter()
        self._ladder_p_snapshot: Counter[tuple[str, str]] = Counter()
        self._summary_ladder: Counter[str] = Counter()
        # What each tier holds by kind (`_tier_contents`): (tier, kind) ->
        # blocks, RAM blocks the GPU caches too by kind, RAM blocks no
        # eviction ranks (pages, states), and when each was counted.
        self.tier_blocks: dict[tuple[str, str], int] = {}
        self.duplicate_blocks: dict[str, int] = {}
        self.unranked_ram_blocks = (0, 0)
        self._tier_blocks_at = self._unranked_at = -float("inf")
        # RAM blocks a restore loads at a time when the GPU lacks room for
        # all of it (`_start_chunked`), and the one doing so.
        self._restore_chunk_blocks = restore_chunk_blocks
        self._chunked: _Restore | None = None
        # Requests whose chunked restore gave up: they restore whole, or wait,
        # until admitted (a stalled swap would only stall again).
        self._no_chunk: set[str] = set()
        # Request -> the session it resumes, from its arrival in the queue
        # until it finishes (`_note_arrival`).
        self._arrived: dict[str, BlockHash] = {}
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
        # Sessions RAM's evictions left with no point to resume at, by the
        # kind of eviction that took the last one (`_report_losses`).
        self.sessions_lost: Counter[str] = Counter()
        self._sessions_lost_snapshot: Counter[str] = Counter()
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
        # GPU blocks that left with no copy in RAM though a lookup could
        # restore them, by what they held (`_note_uncopied`).
        self.uncopied_kinds: Counter[str] = Counter()
        self._restores: dict[str, _Restore] = {}
        # Loads by restore (its whole session, or a chunk of it).
        self._unsent: list[
            tuple[_Restore, list[tuple[KVCacheBlock, KVCacheBlock]]]
        ] = []
        self._restores_by_event: dict[
            int, list[tuple[_Restore, list[tuple[KVCacheBlock, KVCacheBlock]]]]
        ] = {}
        self._load_event_pending_counts: dict[int, int] = {}
        self._loading_cpu: set[int] = set()
        self._num_pinned = 0
        # Request -> when it first waited for room to restore into. The first
        # of them in the queue (its id and queue position), and the RAM blocks
        # it would load, pinned so that making room does not evict them.
        self._waiting_since: dict[str, float] = {}
        # Of that wait, since when the request is held back behind another
        # restore and the seconds it was (they do not count towards
        # `_MAX_RESTORE_WAIT_S`), and its waiting steps by reason.
        self._held_back_since: dict[str, float] = {}
        self._wait_credit: dict[str, float] = {}
        self._wait_steps: dict[str, Counter[str]] = {}
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
                abandon_gap_multiplier=abandon_gap_multiplier,
                abandon_floor_s=abandon_floor_s,
                leaving=self.leaving_copies,
                dropped=self.drop_leaving,
                ladder_ranking=ram_ladder_ranking,
                may_drop_point=self._may_drop_point,
            )
            self.cpu_block_pool.eviction_policy = self.ram_policy
            # The policy ranks RAM; a GPU eviction does not refresh it.
            self.refresh_cpu_on_gpu_eviction = False
        self._ram_evictions_snapshot = RamEvictionStats()

    # ------------------------------------------------------------------
    # Binding

    def bind_kv_cache_manager(self, kv_cache_manager: "KVCacheManager") -> None:
        self._kv_cache_manager = kv_cache_manager

    def bind_first_chunk(
        self,
        first_chunk: Callable[["Request", int, int, tuple[int, ...]], tuple[int, int]],
    ) -> None:
        self._first_chunk = first_chunk

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
        lazy walk copies ahead (`_target_free` blocks). RAM's copy of such a
        block only mirrors it (`RamGreedyDualSizeFrequency._take_duplicates`);
        the walk copies the block again once it reaches that end."""
        pool = self._gpu_block_pool
        if pool is None:
            return lambda key: False
        policy = pool.eviction_policy
        order = (
            policy.peek_eviction_order(self._target_free)
            if policy is not None
            else pool.free_block_queue.iter_blocks_after(None)
        )
        leaving = {b.block_id for b in itertools.islice(order, self._target_free)}
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
        self._index_stored(stored, cpu_block_ids)

    def _index_stored(self, units: list[Any], cpu_block_ids: list[int]) -> None:
        """RAM now holds the copies in `cpu_block_ids` under their keys: its
        policy values their owners again, and the GPU's exact-end states
        among `units` whose blocks are now all in RAM are indexed. Shared by
        the stores that landed and the inline spills (`before_reuse`)."""
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
        for entry in units:
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

    def _frontier_entries(self, block: KVCacheBlock) -> Iterator[Any]:
        """The GPU's exact-end states the cached `block` carries a key of."""
        checkpoints = self._checkpoints
        if checkpoints is None or block.block_hash is None:
            return
        entries = checkpoints.frontiers.entries
        if not entries:
            return
        pool = self._gpu_block_pool
        assert pool is not None
        for key in (
            block.block_hash,
            *pool.cached_block_hashes_by_block.get(block.block_id, ()),
        ):
            entry = entries.get(get_block_hash(key))
            if entry is not None:
                yield entry

    def _frontier_gaps(self, entry: Any) -> tuple[list[KVCacheBlock], bool]:
        """The blocks of the GPU's exact-end state `entry` RAM does not hold
        under its key, and whether one of them is in use: it may not be
        written yet (a gather the next step runs), so RAM cannot take the
        state whole now."""
        cpu = self.cpu_block_pool.cached_block_hash_to_block
        missing = [
            block
            for group_id, block in entry.blocks.items()
            if cpu.get_one_block(make_block_hash_with_group_id(entry.key, group_id))
            is None
        ]
        return missing, any(block.ref_cnt for block in missing)

    def leaves_last(self, block: KVCacheBlock) -> bool:
        """SpillGuard (inline spills): a block of an exact-end state RAM
        cannot take whole yet, a block it lacks of it being in use. The GPU
        drops such a state with the first of its blocks to leave, uncopied
        (`LazySpillStats.inline_units_lost`); behind the other free blocks
        its own usually outlast that use."""
        policy = self.ram_policy
        for entry in self._frontier_entries(block):
            if policy is not None and policy.disowned(
                make_block_hash_with_group_id(entry.key, g) for g in entry.blocks
            ):
                # Never copied (`_frontier_incomplete`).
                continue
            if self._frontier_gaps(entry)[1]:
                return True
        return False

    def _reuse_units(
        self, blocks: list[KVCacheBlock]
    ) -> list[tuple[Any, list[BlockHashWithGroupId], list[KVCacheBlock] | None]]:
        """Inline spills: the GPU's exact-end states a leaving block carries
        a key of. The GPU drops such a state whole with any of its blocks, so
        RAM takes the blocks it lacks of it now, those still in the free
        queue too, or the state is lost (None: one of them is in use). Not a
        state only retired sessions resumed at, nor one with a block RAM
        would not keep."""
        policy = self.ram_policy
        units: list[
            tuple[Any, list[BlockHashWithGroupId], list[KVCacheBlock] | None]
        ] = []
        seen: set[BlockHash] = set()
        for block in blocks:
            for entry in self._frontier_entries(block):
                if entry.key in seen:
                    continue
                seen.add(entry.key)
                keys = [
                    make_block_hash_with_group_id(entry.key, g) for g in entry.blocks
                ]
                if policy is not None and policy.disowned(keys):
                    continue
                missing, in_use = self._frontier_gaps(entry)
                if in_use:
                    units.append((entry, keys, None))
                elif all(self._restorable(member) for member in missing):
                    units.append((entry, keys, missing))
        return units

    def _free_ram_pins(self) -> bool:
        """Inline spills: the waiting restore's pins on the RAM blocks it
        would load, as `_relieve_spill_stall` gives them up. A chunked
        restore keeps its own: the hook may run inside its loop
        (`_advance_chunked`)."""
        if not self._head_pins:
            return False
        self._unpin_head()
        self.restore_stats.unpinned_for_spill += 1
        return True

    def _reuse_cost(
        self, num_blocks: int, pinned: Collection[KVCacheBlock] = ()
    ) -> tuple[int, int]:
        """Inline spills: of the next `num_blocks` GPU blocks a restore would
        take, in the free queue's order (not `pinned`, which it holds first),
        those that need a copy in RAM and those RAM holds a copy of."""
        pool = self._gpu_block_pool
        assert pool is not None
        held = {block.block_id for block in pinned}
        need = copied = taken = 0
        for block in pool.free_block_queue.iter_blocks_after(None):
            if taken == num_blocks:
                break
            if block.block_id in held:
                continue
            taken += 1
            if self.needs_spill(block):
                need += 1
            elif block.block_hash is not None:
                copied += 1
        return need, copied

    def _ram_takes_spill(
        self,
        num_blocks: int,
        pinned: Collection[KVCacheBlock] = (),
        loads: Collection[KVCacheBlock] = (),
    ) -> bool:
        """Inline spills: whether RAM has room for the copies of the next
        `num_blocks` GPU blocks a restore would take (`_reuse_cost`): those
        that need one, against RAM's free blocks without the sources the
        restore has yet to pin (`loads`) and the copies RAM holds of the
        others, pinned while it makes room (`before_reuse`)."""
        need, copied = self._reuse_cost(num_blocks, pinned)
        if not need:
            return True
        room = self.cpu_block_pool.get_num_free_blocks() - copied
        return need <= room - sum(not block.ref_cnt for block in loads)

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
        started = time.perf_counter_ns()
        try:
            return self._lookup(request, num_computed_tokens)
        finally:
            self.restore_stats.lookup_ns += time.perf_counter_ns() - started

    def _lookup(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        request_id = request.request_id
        restore = self._restores.get(request_id)
        if restore is not None:
            if not restore.done:
                self._outcome(request_id, "loading")
                return None, False
            self._outcome(request_id, "landed")
            self._landed_looked_up.add(request_id)
            # From the restore's own point: the scheduler's hit can run past
            # it (pages without their state), which would drop the rungs in
            # between and run the chunk past the room kept for it.
            point = restore.num_tokens - restore.num_tokens % self.block_size
            self._raise_junction(request, point)
            self._set_rungs(request, point)
            return 0, False
        if self._gpu_block_pool is None:
            return 0, False
        plan: _Plan | None = None
        chain: list[_Piece] | None = None
        planned = False
        if self._behind_head(request):
            if not self._pass:
                self.restore_stats.waited_behind += 1
                self._outcome(request_id, "behind")
                self._note_held_back(request_id)
                return None, False
            # Pack admission holds no one behind a waiting restore: what that
            # one waits for is claimed (`claim_against`). "pass" holds only
            # those that would restore too.
            if not request.skip_reading_external_kv:
                plan, chain = self._plan(request, num_computed_tokens)
                planned = True
            if plan is not None and not self._pack:
                self.restore_stats.waited_behind += 1
                self._outcome(request_id, "behind")
                self._note_held_back(request_id)
                return None, False
            if plan is None:
                self.restore_stats.passed_head += 1
                self.restore_stats.passed_head_cold += not num_computed_tokens
        if request.skip_reading_external_kv:
            self._outcome(request_id, "skip")
            return 0, False
        if not planned:
            plan, chain = self._plan(request, num_computed_tokens)
        if plan is None:
            self._stop_waiting(request_id)
            self._outcome(request_id, "gpu_only")
            self._raise_junction(request, num_computed_tokens, chain)
            self._set_rungs(request, num_computed_tokens)
            return 0, False
        started = time.perf_counter_ns()
        try:
            return self._start_restore(request, plan, chain)
        finally:
            self.restore_stats.lookup_room_ns += time.perf_counter_ns() - started

    def _outcome(self, request_id: str, outcome: str) -> None:
        """Count a lookup's outcome; one that leaves the request waiting is
        its reason this step (`WAIT_REASONS`)."""
        self.lookup_outcomes[outcome] += 1
        if outcome in WAIT_REASONS:
            self._last_wait[request_id] = outcome
        else:
            self._last_wait.pop(request_id, None)

    def _rungs(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, ...] | None:
        """The checkpoint ladder's rungs past `num_computed_tokens` (None:
        no ladder, the request keeps its own)."""
        if not self._ladder_blocks or not self.mamba_gids:
            return None
        block_size = self.block_size
        end_block = request.num_prompt_tokens // block_size
        first = num_computed_tokens // block_size
        rungs = []
        distance = self._ladder_blocks
        while end_block - distance > first:
            rungs.append((end_block - distance) * block_size)
            distance *= 2
        return tuple(sorted(rungs))

    def _set_rungs(self, request: "Request", num_computed_tokens: int) -> None:
        """The checkpoint ladder's rungs inside what the request computes
        (`_ladder`): the prefill stops a chunk at each and the GPU keeps its
        state (`Request.retained_boundaries`), which production's sparse Mamba
        retention would not otherwise cache inside a prefill."""
        rungs = self._rungs(request, num_computed_tokens)
        if rungs is not None:
            request.retained_boundaries = rungs

    def _junction(
        self,
        request: "Request",
        num_computed_tokens: int,
        chain: list[_Piece] | None = None,
    ) -> int | None:
        """Where `_raise_junction` would put the request's junction (None:
        where it is). `chain`: the request's page chain, if this lookup found
        it already."""
        started = time.perf_counter_ns()
        try:
            if chain is None:
                chain = self._page_chain(request)
            end = len(chain) * self.block_size
            if end <= max(num_computed_tokens, request.shared_prefix_boundary):
                return None
            key = request.block_hashes[end // self.hash_block_size - 1]
            if all(
                self._find(make_block_hash_with_group_id(key, group_id))
                for group_id in self.mamba_gids
            ):
                return None
            return end
        finally:
            self.restore_stats.lookup_junction_ns += time.perf_counter_ns() - started

    def _raise_junction(
        self,
        request: "Request",
        num_computed_tokens: int,
        chain: list[_Piece] | None = None,
    ) -> None:
        """The scheduler marks a shared prefix's junction from the GPU alone
        (`Request.shared_prefix_boundary`): where the pages run past the
        point the request resumes at, which another request wrote. With
        pages in RAM too, the junction is where the pages either tier holds
        end: the request's prefill stops there, and the state it caches is
        one both tiers keep (`_note_junction`). Not where a state is cached
        already (the request resumes short of it only because restoring it
        would gain too little)."""
        end = self._junction(request, num_computed_tokens, chain)
        if end is not None:
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
        chunked = self._chunked
        if (
            chunked is not None
            and chunked.request_id != request.request_id
            and self._queue_position(request) > chunked.position
        ):
            # It would take the blocks the chunked restore's next chunks wait
            # for, which already hold GPU blocks.
            return True
        head = self._head
        if (
            head is None
            or head[0] == request.request_id
            or self._queue_position(request) <= head[1]
        ):
            return False
        since = self._waiting_since.get(head[0])
        return since is not None and time.monotonic() - since >= _HEAD_PASS_S

    def _stop_waiting(self, request_id: str) -> float:
        """The request no longer waits for room to restore into; how long it
        did (0 if it did not)."""
        since = self._waiting_since.pop(request_id, None)
        waited = 0.0
        if since is not None:
            waited = time.monotonic() - since
            stats = self.restore_stats
            stats.head_waits_over_1s += waited >= 1.0
            stats.head_waits_over_10s += waited >= 10.0
            stats.head_waits_over_60s += waited >= 60.0
        self._waiting_need.pop(request_id, None)
        self._held_back_since.pop(request_id, None)
        self._wait_credit.pop(request_id, None)
        self._wait_steps.pop(request_id, None)
        if self._claim is not None and self._claim[0] == request_id:
            # It has its room, or gave up waiting for it.
            self._claim = None
        if self._head is not None and self._head[0] == request_id:
            self._head = None
            self._unpin_head()
        return waited

    def _note_held_back(self, request_id: str) -> None:
        """A request that waited for room is held back behind another restore
        this step: until it waits for room again, the time is not its own
        wait (`_wait`)."""
        if request_id in self._waiting_since:
            self._held_back_since.setdefault(request_id, time.monotonic())

    def _unpin_head(self) -> None:
        if self._head_pins:
            self.cpu_block_pool.free_blocks(self._head_pins)
            self._head_pins = []

    def _loads_in_flight(self) -> bool:
        return bool(self._restores_by_event or self._unsent)

    def _relieve_spill_stall(self) -> bool:
        """The waiting restore's pins on the RAM blocks it would load: RAM
        has no other block to copy the GPU's into, and the restore waits for
        those GPU blocks. RAM may then evict its blocks by value."""
        if self._head_pins:
            self._unpin_head()
            self.restore_stats.unpinned_for_spill += 1
            return True
        restore = self._chunked
        if restore is not None and restore.pending:
            # Its RAM blocks still to load are pinned: RAM may then evict them.
            self._abort_chunked(restore)
            return True
        return False

    def _skips_held(self, request: "Request", may_skip: bool) -> bool:
        """Pack admission: the scheduler goes on past a request whose
        allocation waits, for copies or for claimed room. The claim keeps the
        room the earliest of them waits for (`claim_against`), and a landed
        restore leaves its own room for it (`claim_against`)."""
        return self._pack and may_skip

    # ------------------------------------------------------------------
    # Pack admission: the claim and the restores' room

    def claim_against(self, request: "Request") -> int:
        """SpillGuard: spill-safe free blocks `request`'s allocation, or
        restore, must leave: the room restores keep for their requests'
        tokens past the point (`_Restore.extra`), but its own; and the
        claimant's room, unless it is the claimant, comes before it in the
        queue (an urgent chat, a preempted request), or has a restore of its
        own. Running requests and those holding blocks leave none: their
        decode never waits on a claim."""
        if not self._pack or (
            request.status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
            or request.num_computed_tokens > 0
        ):
            return 0
        restore = self._restores.get(request.request_id)
        num_blocks = self._reserved - (restore.extra if restore is not None else 0)
        claim = self._claim
        if (
            claim is not None
            and restore is None
            and request.request_id != claim[0]
            and self._queue_position(request) >= claim[1]
        ):
            num_blocks += claim[2]
        # Never negative: a negative claim would let an allocation past the
        # spill guard (`num_spill_safe_free_blocks` counts up to it).
        return max(0, num_blocks)

    def release_reserved(self, request: "Request", num_blocks: int) -> None:
        """SpillGuard: the request's allocation of `num_blocks` goes ahead:
        the room its restore kept for it is the allocation's now."""
        restore = self._restores.get(request.request_id)
        if restore is None or not self._pack:
            return
        stats = self.restore_stats
        restore.kept, restore.used = restore.extra, num_blocks
        if num_blocks > restore.extra:
            stats.extra_short += 1
            stats.extra_short_blocks += num_blocks - restore.extra
        elif num_blocks < restore.extra:
            stats.extra_unused_blocks += restore.extra - num_blocks
        else:
            stats.extra_exact_admissions += 1
        self._reserved -= restore.extra
        restore.extra = 0

    def note_refused(self, request: "Request", reason: str) -> None:
        """SpillGuard: `allocate_slots` refused the request this step, or the
        scheduler scheduled nothing for it. Room others keep splits into the
        claimant's ("claim") and the room restores keep for their requests
        ("reserved")."""
        request_id = request.request_id
        restore = self._restores.get(request_id)
        if reason == "claim":
            own = restore.extra if restore is not None else 0
            if self.claim_against(request) <= self._reserved - own:
                reason = "reserved"
        self._last_wait[request_id] = reason
        if restore is not None and restore.done:
            self.landed_waits[reason] += 1
            self._landed_refused.add(request_id)

    def note_held(self, request: "Request", num_blocks: int) -> None:
        """SpillGuard: a waiting request's allocation waits for copies; the
        earliest such request may be the next claimant."""
        if self._pack and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            self._held_now[request.request_id] = (
                self._queue_position(request),
                num_blocks,
            )
            self._alloc_held_since.setdefault(request.request_id, time.monotonic())

    def note_claim_blocked(self, request: "Request") -> None:
        self.restore_stats.claim_blocked_admissions += 1

    def _update_claim(self, waiting: Iterable["Request"]) -> None:
        """Before the step allocates: the request the others leave room for.
        The chunked restore under way (it holds part of its room already),
        else the earliest queued of the restores waiting for room and the
        allocations the spill guard held last step. Its room is what it
        needs, once it has waited `claim_after_s` (a chunked restore: since
        its last chunk); the walk copies for it first (`_start_restore`)."""
        held, self._held_now = self._held_now, {}
        if not self._pack:
            return
        queued = {r.request_id for r in waiting}
        for request_id in [r for r in self._alloc_held_since if r not in queued]:
            del self._alloc_held_since[request_id]
        claim = None
        since = None
        restore = self._chunked
        if restore is not None and restore.pending:
            claim = (restore.request_id, restore.position, len(restore.pending))
            since = restore.progress_at
        else:
            candidates = []
            head = self._head
            if head is not None and head[0] in queued:
                need = self._waiting_need.get(head[0], 0)
                candidates.append((head[1], head[0], need))
            for request_id, (position, num_blocks) in held.items():
                if request_id in queued and request_id not in self._restores:
                    candidates.append((position, request_id, num_blocks))
            if candidates:
                position, request_id, num_blocks = min(candidates)
                claim = (request_id, position, num_blocks)
                since = self._waiting_since.get(
                    request_id, self._alloc_held_since.get(request_id)
                )
        stats = self.restore_stats
        if (
            claim is not None
            and since is not None
            and time.monotonic() - since < self._claim_after_s
        ):
            stats.claim_deferred_steps += 1
            claim = None
        if claim is not None:
            stats.claim_wait_steps += 1
            if self._claim is None or self._claim[0] != claim[0]:
                stats.claimant_changes += 1
        self._claim = claim

    def observe_queue(self, waiting: Iterable["Request"]) -> None:
        started = time.perf_counter_ns()
        try:
            self._account_landed()
            self._last_wait.clear()
            self._observe_queue(waiting)
            self._update_claim(waiting)
        finally:
            self.restore_stats.observe_queue_ns += time.perf_counter_ns() - started

    def _account_landed(self) -> None:
        """As a scheduling step starts: why each restore that had landed
        before the last one began is still not admitted, when its allocation
        was not refused (`note_refused` counted those): the scheduler looked
        it up but allocated nothing for it, or never reached it."""
        for request_id in self._landed_eligible:
            restore = self._restores.get(request_id)
            if (
                restore is None
                or not restore.done
                or request_id in self._landed_refused
            ):
                continue
            reason = (
                "scheduler" if request_id in self._landed_looked_up else "not_reached"
            )
            self.landed_waits[reason] += 1
        self._landed_eligible = {
            request_id for request_id, r in self._restores.items() if r.done
        }
        self._landed_looked_up.clear()
        self._landed_refused.clear()

    def _observe_queue(self, waiting: Iterable["Request"]) -> None:
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
            if entry is None:
                self._note_arrival(request)
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

    def _note_arrival(self, request: "Request") -> None:
        """A request joined the queue: the session it resumes has one more
        turn gap on record, measured to its arrival (time queued does not
        count), and stays live until it finishes
        (`RamGreedyDualSizeFrequency.arrived`). Gaps are kept with
        abandonment off too: they also time a session's expected return
        (`ram_ladder_ranking`)."""
        policy = self.ram_policy
        request_id = request.request_id
        if policy is None:
            return
        if request_id in self._arrived:
            return
        _, key, extends_end = self._seen_end(request)
        if key is None:
            return
        self._arrived[request_id] = key
        # The policy's clock is monotonic; `arrival_time` is wall time.
        at = time.monotonic() - max(0.0, time.time() - request.arrival_time)
        policy.arrived(key, at, extends_end)

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
        self,
        request: "Request",
        plan: _Plan,
        chain: list[_Piece] | None = None,
    ) -> tuple[int | None, bool]:
        request_id = request.request_id
        pool = self._gpu_block_pool
        assert pool is not None
        loads = [p.block for p in plan.pieces if not p.on_gpu]
        pinned = [p.block for p in plan.pieces if p.on_gpu]
        if any(b.block_id in self._loading_cpu for b in loads):
            # Another request's restore is loading the same RAM blocks.
            self._outcome(request_id, "same_blocks")
            self._note_held_back(request_id)
            return None, False
        num_blocks = len(loads)
        stats = self.restore_stats
        # Pack admission: the restore keeps room for the request's tokens
        # past the point too, so the request is admitted the step its load
        # lands; and leaves the room the other restores keep and the
        # claimant's (none for the claimant).
        extra = chunk_tokens = 0
        if self._pack:
            chunk_tokens, extra = self._estimate_extra(request, plan.num_tokens, chain)
            # Never more than the GPU holds beside the loads and a step's
            # blocks: what does not fit is allocated at admission, as before.
            extra = max(
                0, min(extra, pool.num_gpu_blocks - num_blocks - self._target_free)
            )
        want = num_blocks + extra
        claim = self.claim_against(request)
        behind_claimant = claim > self._reserved
        # The claim's part beyond the room other restores keep: the
        # claimant's.
        claimant_room = max(0, claim - self._reserved)
        # Restores waiting for admission run side by side while the running
        # requests keep a step's blocks free (the lazy walk's target); a
        # restore alone may pin a whole session. A cap of a quarter of the
        # pool let one 60-120-block session restore at a time in production.
        free = pool.get_num_free_blocks() - sum(not b.ref_cnt for b in pinned)
        if (self._num_pinned or claim) and free - want - claim < self._target_free:
            stats.waited_cap += 1
            stats.head_waits_not_copying += 1
            stats.claim_blocked_restores += bool(claim) and (
                not self._num_pinned or free - want >= self._target_free
            )
            by_claimant = bool(claimant_room) and (
                free - want - (claim - claimant_room) >= self._target_free
            )
            return self._wait(
                request,
                loads,
                want,
                "room_claim" if by_claimant else "room_restores",
                plan.num_tokens,
            )
        # Blocks pinned below leave the free queue: they are not safe to take.
        # Once the spill guard lets held allocations go, neither does this one
        # wait for copies (`KVCacheManager.allocate_slots`).
        safe = (
            pool.num_spill_safe_free_blocks(want + claim, exclude=pinned)
            if self.holding or self._inline
            else free
        )
        if self._inline:
            # Inline spills hold nothing back and take the free queue in its
            # own order: a restore goes whole while the pool has the blocks,
            # those it would take cost no more copies than a chunk (any
            # number, with chunks off) and RAM has room for them, its own
            # sources pinned. One that costs more loads in chunks where it
            # can, so RAM seldom makes room for a whole session's copies at
            # once.
            cost = self._reuse_cost(num_blocks, pinned)[0]
            fits = free >= want + claim
            blocked = not (
                fits
                and (not self._chunk_blocks() or cost <= self._chunk_blocks())
                and self._ram_takes_spill(num_blocks, pinned, loads)
            )
        else:
            fits = False
            blocked = safe < want + claim
        if blocked:
            stats.claim_blocked_restores += safe >= want
            # Only one restore holds part of its room: the claimant (pack), or
            # whichever chunks first (legacy).
            if (
                not behind_claimant
                and request_id not in self._no_chunk
                and self._can_chunk(num_blocks, free, loads, extra + claim)
            ):
                return self._start_chunked(
                    request, plan, loads, pinned, extra, chunk_tokens
                )
            # With inline spills it waits for no copy: for the pool's blocks,
            # for the chunked restore's turn, or for a load to land and give
            # RAM its sources back. With the pool's blocks there and none of
            # those under way it goes whole, as it does with the walk once its
            # copies are done: it cannot chunk (RAM is too small to swap a
            # chunk, or the pool lacks a step's blocks beside it), and nothing
            # it could wait for will change that (GPU windows C1 and C1b: such
            # restores waited out `_MAX_RESTORE_WAIT_S` with nothing else
            # running). The copies RAM has no room for are not taken
            # (`unspilled_evicted_blocks`), as for any allocation.
            blocked = not (
                self._inline
                and fits
                and self._chunked is None
                and not self._loads_in_flight()
            )
            stats.started_without_room += not blocked
        if blocked:
            stats.waited_guard += not self._inline
            # Only what copies can free: blocks the pool lacks altogether come
            # back as requests finish, not from the walk.
            if self._pack:
                # Copies for every waiting restore the GPU has room for, so
                # those that fit start side by side next step; the claimant's
                # room stays its own (`claim_against`). (Beside other pinned
                # restores or a claim, the cap above left a step's blocks.)
                fits = free >= want + claim
                need = want - max(0, safe - claim) if fits else 0
            else:
                # With chunks, no more than one: the rest would only evict
                # RAM's blocks while this one waits.
                need = min(want, free) - safe
                if self._chunk_blocks():
                    need = min(need, self._chunk_blocks())
                    if free < want + self._target_free:
                        need = 0
            if self.holding and need > 0:
                self.on_allocation_held(need)
                stats.head_waits_copying += 1
            else:
                stats.head_waits_not_copying += 1
            if safe >= want + claim - claimant_room and claimant_room:
                reason = "room_claim"
            elif safe >= want or self._inline:
                reason = "room_restores"
            else:
                reason = "copies_restore"
            return self._wait(request, loads, want, reason, plan.num_tokens)
        waited_s = self._stop_waiting(request_id)
        pool.touch(pinned)
        if self._inline:
            # Before its GPU blocks: taking them copies what they hold into
            # RAM (`before_reuse`), which must not evict its own sources.
            self.cpu_block_pool.touch(loads)
        gpu_blocks = pool.get_new_blocks(num_blocks)
        if not self._inline:
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
            extra=extra,
            chunk_tokens=chunk_tokens,
            request=request,
            started_at=time.monotonic(),
        )
        self._reserved += extra
        self._restores[request_id] = restore
        self._unsent.append((restore, restore.loads))
        self._loading_cpu.update(b.block_id for b in loads)
        self._num_pinned += restore.num_pinned
        stats.started += 1
        self._note_start()
        self._outcome(request_id, "started")
        self._log_restore_start(restore, plan, waited_s)
        return None, False

    def _log_restore(
        self, event: str, restore: _Restore, message: str, *args: Any
    ) -> None:
        level = self._restore_log_level
        if logger.isEnabledFor(level):
            logger.log(
                level,
                "Tiered KV restore %s for %s at %d tokens: " + message,
                event,
                restore.request_id,
                restore.num_tokens,
                *args,
            )

    def _log_restore_start(
        self, restore: _Restore, plan: _Plan, waited_s: float
    ) -> None:
        if not logger.isEnabledFor(self._restore_log_level):
            return
        self._log_restore(
            "started",
            restore,
            "%d RAM blocks to load, %d on the GPU, %s; GPU alone resumes at %d; "
            "room %d blocks for a first chunk of %d tokens; waited %.0f ms; "
            "%d restores loading, %d blocks kept for others",
            len(restore.pending) + len(restore.loads),
            len(restore.pinned),
            "in chunks" if restore.chunked else "whole",
            plan.gpu_only_tokens,
            restore.extra,
            restore.chunk_tokens,
            1000 * waited_s,
            sum(not r.done for r in self._restores.values()),
            self._reserved - restore.extra,
        )

    def _log_summary(self) -> None:
        """At most every `summary_log_s`, after activity: what restores and
        admissions did since the last line."""
        now = time.monotonic()
        if not self._summary_log_s or now - self._summary_at < self._summary_log_s:
            return
        current, last = self.restore_stats, self._summary_snapshot

        def delta(name: str) -> int:
            return getattr(current, name) - getattr(last, name)

        outcomes = self.lookup_outcomes - self._summary_outcomes
        landed = self.landed_waits - self._summary_landed
        pool = self._gpu_block_pool
        spill = (
            (0, 0, 0, 0)
            if pool is None
            else (
                pool.spill_scan_calls,
                pool.spill_scan_blocks,
                pool.spill_scan_ns,
                pool.spill_reorder_ns,
            )
        )
        scans = spill[0] - self._summary_spill[0]
        ladder = self.ladder_events - self._summary_ladder
        active = delta("started") or sum(outcomes.values())
        if ladder or (self.tier_blocks and active):
            self._log_tiers(ladder)
        if active:
            steps = max(1, self._step - self._summary_step)
            elapsed_ns = max(1, int(1e9 * (now - self._summary_at)))
            connector_ns = delta("lookup_ns") + delta("observe_queue_ns")
            connector_ns += delta("step_ns") + delta("stats_ns")
            admitted = sum(
                delta(f) for f in ("admit_steps_0", "admit_steps_1", "admit_steps_2_4")
            ) + delta("admit_steps_over_4")
            logger.info(
                "Tiered KV, last %.0f s, %d steps: restores %d started (%d in "
                "chunks), %d landed, %d admitted (%d the step they landed), "
                "%d gave up; room for their first chunk exact %d, over %d "
                "blocks, short %d (%d blocks); lookups %s; landed restores "
                "waiting, steps: %s; claim %d steps (%d deferred); connector "
                "%.2f ms per step, %.1f%% of the time (search %.2f, room "
                "%.2f, junction %.2f, queue %.2f, step %.2f, stats %.2f ms per "
                "step); spill scans %d (%.0f blocks each, %.2f ms per step); "
                "allocations held for copies %d",
                now - self._summary_at,
                steps,
                delta("started"),
                delta("chunked"),
                delta("completed"),
                admitted,
                delta("admit_steps_0"),
                delta("gave_up"),
                delta("extra_exact_admissions"),
                delta("extra_unused_blocks"),
                delta("extra_short"),
                delta("extra_short_blocks"),
                dict(outcomes.most_common()) or "{}",
                dict(landed.most_common()) or "{}",
                delta("claim_wait_steps"),
                delta("claim_deferred_steps"),
                connector_ns / 1e6 / steps,
                100 * connector_ns / elapsed_ns,
                delta("lookup_search_ns") / 1e6 / steps,
                delta("lookup_room_ns") / 1e6 / steps,
                delta("lookup_junction_ns") / 1e6 / steps,
                delta("observe_queue_ns") / 1e6 / steps,
                delta("step_ns") / 1e6 / steps,
                delta("stats_ns") / 1e6 / steps,
                scans,
                (spill[1] - self._summary_spill[1]) / max(1, scans),
                (spill[2] - self._summary_spill[2] + spill[3] - self._summary_spill[3])
                / 1e6
                / steps,
                self.lazy_spill_stats.held_allocations - self._summary_held_allocations,
            )
        self._summary_at, self._summary_step = now, self._step
        self._summary_snapshot = replace(current)
        self._summary_outcomes = Counter(self.lookup_outcomes)
        self._summary_landed = Counter(self.landed_waits)
        self._summary_spill = spill
        self._summary_held_allocations = self.lazy_spill_stats.held_allocations
        self._summary_ladder = Counter(self.ladder_events)

    def _log_tiers(self, ladder: Counter[str]) -> None:
        """What each tier holds (`_tier_contents`), the distinct blocks they
        add up to, and ladder events since the last summary."""
        counts, duplicates = self.tier_blocks, self.duplicate_blocks

        def held(tier: str) -> dict[str, int]:
            return {k: counts[tier, k] for k in TIER_BLOCK_KINDS if counts[tier, k]}

        cached = {
            tier: sum(n for (t, k), n in counts.items() if t == tier and k != "free")
            for tier in ("gpu", "ram")
        }
        pages, states = self.unranked_ram_blocks
        logger.info(
            "Tiered KV tiers: GPU %s; RAM %s; distinct %d blocks (GPU %d + RAM "
            "%d - %d in both: %s); RAM no eviction ranks %d pages, %d states; "
            "ladder %s",
            held("gpu"),
            held("ram"),
            cached["gpu"] + cached["ram"] - sum(duplicates.values()),
            cached["gpu"],
            cached["ram"],
            sum(duplicates.values()),
            duplicates,
            pages,
            states,
            dict(ladder.most_common()) or "{}",
        )

    def _note_start(self) -> None:
        stats = self.restore_stats
        self._starts_this_step += 1
        stats.starts_per_step_max = max(
            stats.starts_per_step_max, self._starts_this_step
        )
        stats.multi_start_steps += self._starts_this_step == 2
        loading = sum(not r.done for r in self._restores.values())
        stats.concurrent_restores_max = max(stats.concurrent_restores_max, loading)

    def _estimate_extra(
        self,
        request: "Request",
        num_tokens: int,
        chain: list[_Piece] | None = None,
    ) -> tuple[int, int]:
        """The tokens and blocks of the request's first chunk past a point of
        `num_tokens`, as the scheduler schedules it (`Scheduler.first_chunk`,
        vLLM's own block count): past the rungs and the junction the request
        will have once restored there (`_lookup`). The room is kept as a
        count, so an estimate too high only keeps others out a step, one too
        low makes the request wait at admission (`release_reserved` counts
        both). The watermark is not kept: admission checks it on its own.
        `chain`: the request's page chain, if this lookup found it."""
        first_chunk = self._first_chunk
        if first_chunk is None:
            return 0, 0
        started = time.perf_counter_ns()
        point = num_tokens - num_tokens % self.block_size
        junction = self._junction(request, point, chain)
        rungs = self._rungs(request, point)
        tokens, blocks = first_chunk(
            request,
            num_tokens,
            request.shared_prefix_boundary if junction is None else junction,
            request.retained_boundaries if rungs is None else rungs,
        )
        stats = self.restore_stats
        if not tokens:
            stats.extra_zero_chunk += 1
        stats.estimate_ns += time.perf_counter_ns() - started
        return tokens, blocks

    def _reestimate_extra(self, restore: _Restore) -> None:
        """As the restore's load lands: the room for its request's first chunk
        again, as the session's blocks around it may have changed since it
        started (its chain, a junction, a duplicate key). It grows only into
        free blocks others do not keep; what it cannot is counted short at
        admission."""
        request = restore.request
        pool = self._gpu_block_pool
        if not self._pack or request is None or pool is None:
            return
        _, estimate = self._estimate_extra(request, restore.num_tokens)
        estimate = max(
            0,
            min(estimate, pool.num_gpu_blocks - restore.num_pinned - self._target_free),
        )
        stats = self.restore_stats
        if estimate > restore.extra:
            room = pool.get_num_free_blocks() - self._reserved - self._target_free
            grow = min(estimate - restore.extra, max(0, room))
            if grow:
                stats.extra_regrown += 1
                stats.extra_regrown_blocks += grow
                restore.extra += grow
                self._reserved += grow
        elif estimate < restore.extra:
            stats.extra_shrunk += 1
            stats.extra_shrunk_blocks += restore.extra - estimate
            self._reserved -= restore.extra - estimate
            restore.extra = estimate

    def _kept_copies(self) -> int:
        """Free RAM blocks RAM never takes for the walk's copies: copies of
        GPU blocks about to leave, in LRU order. RAM GDSF takes those too, by
        value, last (`leaving_copies`), as the walk's budget counts them."""
        if self.cpu_block_pool.eviction_policy is not None:
            return 0
        return len(self.leaving_copies())

    def _chunk_blocks(self) -> int:
        """RAM blocks a chunked restore loads at a time: at most a quarter of
        RAM, which must take the GPU's blocks it spills meanwhile."""
        return min(self._restore_chunk_blocks, self.cpu_block_pool.num_gpu_blocks // 4)

    def _can_chunk(
        self, num_blocks: int, free: int, loads: list[KVCacheBlock], extra: int = 0
    ) -> bool:
        """Whether a restore of `num_blocks` RAM blocks the GPU has no safe
        room for yet may load in chunks: none other does (two partial
        restores could each hold blocks the other waits for), the GPU has
        room for it and its `extra` blocks beside the running requests' step
        (`free` excludes the pieces it holds already), and RAM, its sources
        pinned, can still take one chunk of the GPU's blocks."""
        k = self._chunk_blocks()
        if not k or num_blocks <= k or self._chunked is not None:
            return False
        if free - num_blocks - extra < self._target_free:
            return False
        cpu_free = (
            self.cpu_block_pool.get_num_free_blocks()
            - sum(not b.ref_cnt for b in loads)
            - self._kept_copies()
        )
        return cpu_free >= k

    def _start_chunked(
        self,
        request: "Request",
        plan: _Plan,
        loads: list[KVCacheBlock],
        pinned: list[KVCacheBlock],
        extra: int = 0,
        chunk_tokens: int = 0,
    ) -> tuple[int | None, bool]:
        """Restore in chunks: a swap. The GPU's blocks spilled for each chunk
        land in RAM blocks the previous chunk's load freed: once loaded, the
        RAM copy of a GPU block the restore holds is the first block RAM
        gives up (`RamGreedyDualSizeFrequency._take_duplicates`). Taking the
        whole session's room up front instead spilled as many GPU blocks
        while its RAM blocks were still pinned for the load, and RAM evicted
        other sessions for them (window 5: 80 restores of 58 blocks on
        average). The pages load first, the point's own blocks last together
        (`_advance_chunked`); the request waits until all landed. Under pack
        admission it keeps the room for its tokens past the point (`extra`)
        from the start, as a count the chunks leave alone too."""
        pool = self._gpu_block_pool
        assert pool is not None
        request_id = request.request_id
        waited_s = self._stop_waiting(request_id)
        pool.touch(pinned)
        self.cpu_block_pool.touch(loads)
        tail = [p.block for p in plan.pieces[plan.num_chain :] if not p.on_gpu]
        restore = _Restore(
            request_id,
            plan.num_tokens,
            [],
            pinned,
            plan.frontier,
            plan.state,
            pin_timeout_s=max(_PIN_TIMEOUT_S, self._restore_wait_s(plan)),
            chunked=True,
            pending=list(loads),
            num_tail=len(tail),
            position=self._queue_position(request),
            progress_at=time.monotonic(),
            extra=extra,
            chunk_tokens=chunk_tokens,
            request=request,
            started_at=time.monotonic(),
        )
        self._reserved += extra
        self._restores[request_id] = restore
        self._chunked = restore
        if self._pack and not self._claim_after_s:
            # It holds part of its room from now on: the rest is claimed (with
            # `claim_after_s`, once a chunk waits that long: `_update_claim`).
            self._claim = (request_id, restore.position, len(loads))
        self._loading_cpu.update(b.block_id for b in loads)
        self._num_pinned += len(pinned)
        self.restore_stats.started += 1
        self.restore_stats.chunked += 1
        self._note_start()
        self._outcome(request_id, "started_chunked")
        self._log_restore_start(restore, plan, waited_s)
        return None, False

    def _advance_chunked(self) -> None:
        """Before the walk: the chunked restore takes GPU blocks for its next
        chunks while the spill guard lets it, leaving the running requests a
        step's blocks (`_target_free`) and the room restores keep for their
        requests' tokens past the point (`_reserved`, its own too), and queues
        their loads. What its next chunk still lacks the walk spills this
        step: under pack admission as copies ahead (`_copy_ahead`), which hold
        no allocation back, so the next chunk's copies run while this one
        loads. It gives up the rest after `_CHUNK_STALL_S` without a chunk.
        With inline spills a chunk waits for no copies (its blocks' are taken
        in its own step): only for the step's blocks and the room kept, for
        RAM to have room for those copies, and for its loads to land once
        `_INLINE_CHUNKS_UNACKED` chunks are out; and it does not give up
        while one is."""
        restore = self._chunked
        if restore is None or not restore.pending:
            return
        pool = self._gpu_block_pool
        assert pool is not None
        k = max(1, self._chunk_blocks())
        now = time.monotonic()
        kept = self._reserved
        stats = self.restore_stats
        # Chunks taken whose loads have not landed: those sent, and below
        # those this step sends.
        unacked = restore.in_flight
        while restore.pending:
            pages = len(restore.pending) - restore.num_tail
            n = min(k, pages) if pages > 0 else len(restore.pending)
            free = pool.get_num_free_blocks()
            if self._inline:
                if (
                    free - n - kept < self._target_free
                    or unacked >= _INLINE_CHUNKS_UNACKED
                ):
                    break
                if not self._ram_takes_spill(n):
                    if self._loads_in_flight():
                        # RAM gets its sources back as it lands.
                        break
                    # Nothing else frees RAM but the waiting restore's pins:
                    # the chunk goes, and the copies RAM still has no room
                    # for are not taken (`unspilled_evicted_blocks`).
                    self._free_ram_pins()
                    stats.started_without_room += not self._ram_takes_spill(n)
            else:
                safe = (
                    pool.num_spill_safe_free_blocks(n + kept) if self.holding else free
                )
                if safe < n + kept or free - n - kept < self._target_free:
                    need = min(n + kept, free - self._target_free) - safe
                    if self.holding and need > 0:
                        if self._pack:
                            self._copy_ahead += need
                            stats.copy_ahead_blocks += need
                        else:
                            self.on_allocation_held(need)
                    break
            sources, restore.pending = restore.pending[:n], restore.pending[n:]
            pairs = list(zip(pool.get_new_blocks(n), sources))
            restore.loads.extend(pairs)
            self._num_pinned += n
            self._unsent.append((restore, pairs))
            restore.progress_at = now
            stats.chunks += 1
            unacked += 1
            stats.chunks_unacked_max = max(stats.chunks_unacked_max, unacked)
        if (
            restore.pending
            and now - restore.progress_at > _CHUNK_STALL_S
            and not (self._inline and restore.in_flight)
        ):
            self._abort_chunked(restore)

    def _abort_chunked(self, restore: _Restore, count: bool = True) -> None:
        """Give up the rest of a chunked restore. Its RAM blocks not loaded go
        back to RAM's order, and the chunks loaded stay cached on the GPU once
        any in flight lands (`_release`): the request's next lookup resumes
        from them."""
        if restore.pending:
            self._loading_cpu.difference_update(b.block_id for b in restore.pending)
            self.cpu_block_pool.free_blocks(restore.pending)
            restore.pending = []
        # Chunks taken this step but not sent: nothing loads them.
        unsent = [pairs for r, pairs in self._unsent if r is restore]
        if unsent:
            self._unsent = [(r, pairs) for r, pairs in self._unsent if r is not restore]
            for pairs in unsent:
                self._loading_cpu.difference_update(c.block_id for _, c in pairs)
                self.cpu_block_pool.free_blocks([c for _, c in pairs])
        if self._chunked is restore:
            self._chunked = None
        if self._restores.get(restore.request_id) is restore:
            del self._restores[restore.request_id]
        restore.aborted = True
        if count:
            self.restore_stats.chunked_aborted += 1
            # Pack admission chunks the claimant's restore again: a session
            # larger than the free pool has no other way in. Nor has one
            # larger than a chunk with inline spills, where nothing copies
            # ahead for a whole restore.
            if not self._pack and not self._inline:
                self._no_chunk.add(restore.request_id)
        if not restore.in_flight:
            self._release(restore)

    def _wait(
        self,
        request: "Request",
        loads: list[KVCacheBlock],
        want: int = -1,
        reason: str = "room_restores",
        num_tokens: int = 0,
    ) -> tuple[int | None, bool]:
        """Wait a step for room to restore into (`_MAX_RESTORE_WAIT_S`): room
        for `want` blocks (its loads, and under pack admission the blocks for
        its tokens past the point), for `reason` (`WAIT_REASONS`). The first
        such request in the queue pins the RAM blocks it would load; under
        legacy admission it holds the requests behind it back, under pack it
        may claim its room (`_update_claim`). The limit counts the steps it
        waited for room itself, not those it was held back behind another
        restore: GPU window 5 (2026-11-23) gave up three restores at once
        that way, 287k tokens RAM held whole. `num_tokens`: the point it
        would resume at."""
        request_id = request.request_id
        if want < 0:
            want = len(loads)
        now = time.monotonic()
        since = self._waiting_since.setdefault(request_id, now)
        held_back = self._held_back_since.pop(request_id, None)
        credit = self._wait_credit.get(request_id, 0.0)
        if held_back is not None:
            credit += now - max(held_back, since)
            self._wait_credit[request_id] = credit
        steps = self._wait_steps.setdefault(request_id, Counter())
        if now - since - credit >= _MAX_RESTORE_WAIT_S:
            logger.warning(
                "Restore for %s gave up after %.0f s waiting for room, %.0f s "
                "more held back behind other restores: RAM resumes it at %d "
                "tokens (%d blocks to load); its waiting steps by reason: %s",
                request_id,
                now - since - credit,
                credit,
                num_tokens,
                len(loads),
                dict(steps),
            )
            self._stop_waiting(request_id)
            self.restore_stats.gave_up += 1
            self._outcome(request_id, "gave_up")
            return 0, False
        steps[reason] += 1
        self._outcome(request_id, reason)
        self._waiting_need[request_id] = want
        head = self._head
        position = self._queue_position(request)
        if head is None or head[0] == request_id or position < head[1]:
            self._unpin_head()
            self._head = (request_id, position)
            # The GPU blocks it waits for may need copies in RAM first: it
            # pins its own only while the rest of RAM still takes those
            # (copies of blocks about to leave take nothing: `leaving_copies`).
            cpu_pool = self.cpu_block_pool
            free = cpu_pool.get_num_free_blocks() - self._kept_copies()
            if free - len(loads) >= want:
                cpu_pool.touch(loads)
                self._head_pins = list(loads)
        return None, False

    def _plan(
        self, request: "Request", gpu_hit: int
    ) -> tuple[_Plan | None, list[_Piece]]:
        """The deepest point `request` can resume at with blocks from either
        tier, when RAM holds some of them and it resumes deeper than the GPU
        alone; else None. And its page chain (`_page_chain`), for this
        lookup's junction. `gpu_hit`: the GPU's own hit, block-aligned."""
        started = time.perf_counter_ns()
        stats = self.restore_stats
        try:
            chain = self._page_chain(request)
            plan, visited, units = self._search(
                request, chain, gpu_hit, self._bounded_search
            )
            stats.search_calls += 1
            stats.search_points += visited
            stats.search_units += units
            if self._verify_search and self._bounded_search:
                whole, whole_visited, whole_units = self._search(
                    request, chain, gpu_hit, False
                )
                stats.search_points_whole += whole_visited
                stats.search_units_whole += whole_units
                if (plan is None) != (whole is None) or (
                    plan is not None
                    and whole is not None
                    and (
                        plan.num_tokens != whole.num_tokens
                        or [p.block.block_id for p in plan.pieces]
                        != [p.block.block_id for p in whole.pieces]
                    )
                ):
                    stats.search_mismatches += 1
                    logger.warning(
                        "Bounded resume search disagrees with the whole walk "
                        "for request %s: %s vs %s tokens",
                        request.request_id,
                        None if plan is None else plan.num_tokens,
                        None if whole is None else whole.num_tokens,
                    )
                elif (
                    plan is not None
                    and whole is not None
                    and plan.gpu_only_tokens != whole.gpu_only_tokens
                ):
                    stats.search_gpu_only_differs += 1
            return plan, chain
        finally:
            stats.lookup_search_ns += time.perf_counter_ns() - started

    def _search(
        self,
        request: "Request",
        chain: list[_Piece],
        gpu_hit: int,
        bounded: bool,
    ) -> tuple[_Plan | None, int, int]:
        """`_plan`'s search, the points it visited and the hash units it
        scanned. Points come deepest first. The deepest, if RAM holds part of
        it, is worth a restore when the deepest point the GPU alone holds is
        `_MIN_RESTORE_GAIN_TOKENS` short of it or more. Bounded, the search
        stops at that line, at a point below it or, with none, before the
        units below it (`_Bound`): nothing below changes that, so the plan is
        the whole walk's. Only where the GPU alone would resume, which sets
        the pin timeout, is then taken from `gpu_hit`, below the line."""
        deepest = None
        visited = 0
        gpu_only = None
        bound = _Bound()
        for plan in self._resume_points(request, chain, bound):
            visited += 1
            if deepest is None:
                if plan.on_gpu_only:
                    return None, visited, bound.units
                deepest = plan
                if bounded:
                    bound.floor = deepest.num_tokens - _MIN_RESTORE_GAIN_TOKENS
                continue
            if plan.on_gpu_only:
                gpu_only = plan.num_tokens
                break
            if bounded and plan.num_tokens <= bound.floor:
                gpu_only = min(gpu_hit, bound.floor)
                break
        if deepest is None:
            return None, visited, bound.units
        if gpu_only is None:
            gpu_only = min(gpu_hit, bound.floor) if bound.stopped else 0
        if deepest.num_tokens - gpu_only < _MIN_RESTORE_GAIN_TOKENS:
            return None, visited, bound.units
        deepest.gpu_only_tokens = gpu_only
        return deepest, visited, bound.units

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

    def _resume_points(
        self,
        request: "Request",
        chain: list[_Piece] | None = None,
        bound: _Bound | None = None,
    ) -> Iterator[_Plan]:
        """Points `request` can resume at with blocks from either tier,
        deepest first: exact-end states (GPU frontiers and RAM ones), then
        states at hash-unit boundaries. With sliding-window groups, only
        block boundaries whose windows either tier holds (`_window_pieces`):
        the GPU's lookup after the restore finds a hit there only with them.
        `chain`: its page chain, if found already. `bound`: the caller's
        line, set as it goes; the units scanned."""
        windows = self._window_blocks
        if windows and not self._windows_restorable:
            return
        find = self._find
        unit = self.hash_block_size
        hashes = request.block_hashes
        max_length = request.num_tokens - 1
        if chain is None:
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
            if bound is not None:
                # Unit h's points (its exact ends, its boundary) have fewer
                # than (h + 1) * unit tokens: none past the line, none after.
                if bound.floor >= 0 and (h + 1) * unit <= bound.floor + 1:
                    bound.stopped = True
                    return
                bound.units += 1
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
                yield _Plan(
                    num_tokens, pieces, state=(block_hash, num_tokens), num_chain=k
                )

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
                    yield _Plan(num_tokens, chain[:k] + pieces, num_chain=k)
                    continue
                ram_unit = self.ram_frontiers.entries[expected]
                pieces = []
                for group_id in ram_unit.groups:
                    piece = find(make_block_hash_with_group_id(expected, group_id))
                    if piece is None or piece.on_gpu:
                        break
                    pieces.append(piece)
                else:
                    yield _Plan(
                        num_tokens, chain[:k] + pieces, frontier=ram_unit, num_chain=k
                    )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> SimpleCPUOffloadMetadata:
        started = time.perf_counter_ns()
        try:
            return self._build_connector_meta(scheduler_output)
        finally:
            self.restore_stats.step_ns += time.perf_counter_ns() - started

    def _build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> SimpleCPUOffloadMetadata:
        self._advance_chunked()
        meta = super().build_connector_meta(scheduler_output)
        # After the walk's RAM evictions, before the prunes: the GPU prunes a
        # dropped ladder point in this step.
        self._drain_dropped_points()
        self._report_losses()
        self._apply_prunes()
        self._expire_pins()
        self._step += 1
        self._starts_this_step = 0
        self._log_summary()
        if not self._unsent:
            return meta
        assert meta.load_event < 0, "the tiered connector loads only restores"
        event = self._load_event_counter
        self._load_event_counter += 1
        for restore, pairs in self._unsent:
            restore.event = event
            restore.in_flight += 1
            for gpu_block, cpu_block in pairs:
                meta.load_gpu_blocks.append(gpu_block.block_id)
                meta.load_cpu_blocks.append(cpu_block.block_id)
        self._restores_by_event[event] = self._unsent
        self._unsent = []
        meta.load_event = event
        return meta

    def on_evicted(self, block: KVCacheBlock) -> None:
        """SpillGuard: also note a GPU block leaving with no copy in RAM for
        the session-loss check (`_report_losses`): a session can lose its
        last point on the GPU's side too (a state the guard lets go)."""
        policy, pool = self.ram_policy, self._gpu_block_pool
        if policy is not None and pool is not None and block.block_hash is not None:
            keys = (
                block.block_hash,
                *pool.cached_block_hashes_by_block.get(block.block_id, ()),
            )
            cpu = self.cpu_block_pool.cached_block_hash_to_block
            if any(cpu.get_one_block(key) is None for key in keys):
                policy.note_evicting(keys, "gpu")
        super().on_evicted(block)

    def _note_uncopied(self, block: KVCacheBlock) -> None:
        """Count the block by what it held (`uncopied_kinds`): an attention
        page, a state at a block boundary that a session resumes at or
        another, a state inside a block that is a live exact-end state's or
        another (a prompt-tail state, or an exact-end state the GPU has
        already let go)."""
        assert block.block_hash is not None
        if get_group_id(block.block_hash) not in self._mamba_gid_set:
            kind = "page"
        else:
            num_tokens = block.block_hash_num_tokens
            key = get_block_hash(block.block_hash)
            if num_tokens is None or not num_tokens % self.block_size:
                kind = "boundary_kept" if key in self._kept else "boundary_other"
            elif any(True for _ in self._frontier_entries(block)):
                kind = "exact_end"
            else:
                kind = "in_block_other"
        self.uncopied_kinds[kind] += 1

    def _report_losses(self) -> None:
        """Log each session the walk's RAM evictions this step left with no
        point of its own to resume at (its next turn re-reads it whole), and
        count it by the most telling kind of eviction that took its blocks."""
        if self.ram_policy is None:
            return
        for loss in self.ram_policy.settle():
            kind = next((k for k in _LOSS_KIND_ORDER if k in loss.kinds), loss.kinds[0])
            self.sessions_lost[kind] += 1
            logger.info(
                "RAM lost session %s: %d tokens, resumable at %d tokens before "
                "and %d after; %d of its blocks evicted (%s); idle %.0f s; "
                "%d restores in flight",
                bytes(loss.key)[:6].hex(),
                loss.end,
                loss.reach_before,
                loss.reach_after,
                loss.blocks,
                "/".join(loss.kinds),
                loss.idle_s,
                len(self._restores),
            )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        started = time.perf_counter_ns()
        try:
            self._update_connector_output(connector_output)
        finally:
            self.restore_stats.step_ns += time.perf_counter_ns() - started

    def _update_connector_output(self, connector_output: KVConnectorOutput) -> None:
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
        for restore, pairs in self._restores_by_event.pop(event, []):
            cpu_blocks = [c for _, c in pairs]
            self._loading_cpu.difference_update(c.block_id for c in cpu_blocks)
            restore.in_flight -= 1
            if not restore.reset:
                # A chunk's blocks are cached at once: the restore holds them,
                # so their RAM copies are the first RAM gives up (a swap).
                for gpu_block, cpu_block in pairs:
                    self._adopt_block(gpu_block, cpu_block)
                self.restore_stats.restored_blocks += len(pairs)
                self._interval_load_blocks_completed += len(pairs)
            # The RAM copies were just used: the most recent in RAM.
            cpu_pool.free_blocks(cpu_blocks)
            if self._inline and restore.pending:
                # Inline spills pace a chunked restore by its loads landing
                # (`_INLINE_CHUNKS_UNACKED`): the stall clock starts here.
                restore.progress_at = time.monotonic()
            if restore.pending or restore.in_flight:
                continue
            # Every chunk landed.
            if not restore.reset and not restore.aborted:
                self._register_units(restore)
                self.restore_stats.completed += 1
            restore.done = True
            restore.done_at = time.monotonic()
            restore.done_step = self._step
            if self._chunked is restore:
                self._chunked = None
            if restore.orphaned or restore.reset or restore.aborted:
                self._release(restore)
                continue
            self._reestimate_extra(restore)
            self._log_restore(
                "landed",
                restore,
                "after %.0f ms, room %d blocks",
                1000 * (restore.done_at - restore.started_at),
                restore.extra,
            )

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
        self._no_chunk.discard(request.request_id)
        self._alloc_held_since.pop(request.request_id, None)
        if self._claim is not None and self._claim[0] == request.request_id:
            self._claim = None
        restore = self._restores.get(request.request_id)
        if restore is not None and restore.done:
            # The request holds its blocks now, the step after its load landed
            # at the earliest (`done_step`: a step's loads land after it).
            steps = self._step - restore.done_step
            stats = self.restore_stats
            if steps <= 0:
                stats.admit_steps_0 += 1
            elif steps == 1:
                stats.admit_steps_1 += 1
            elif steps <= 4:
                stats.admit_steps_2_4 += 1
            else:
                stats.admit_steps_over_4 += 1
            self._log_restore(
                "admitted",
                restore,
                "%d steps after landing, room %d blocks for %d tokens, took %d",
                steps,
                restore.kept,
                restore.chunk_tokens,
                restore.used,
            )
            self._release(restore)
            self._admitted_from_ram.add(request.request_id)
            self._admitted_points[request.request_id] = restore.num_tokens

    def _release(self, restore: _Restore) -> None:
        if self._restores.get(restore.request_id) is restore:
            del self._restores[restore.request_id]
        pool = self._gpu_block_pool
        assert pool is not None
        blocks = restore.pinned + [g for g, _ in restore.loads]
        restore.pinned, restore.loads = [], []
        if blocks:
            # The session's deepest blocks go first, its prefix last.
            pool.free_blocks(reversed(blocks))
        self._num_pinned -= len(blocks)
        if restore.extra:
            # Kept for an admission that did not come (`release_reserved`).
            self.restore_stats.extra_released_unadmitted += restore.extra
            self._reserved -= restore.extra
            restore.extra = 0
        if restore.pending:
            self._loading_cpu.difference_update(b.block_id for b in restore.pending)
            self.cpu_block_pool.free_blocks(restore.pending)
            restore.pending = []
        if self._chunked is restore:
            self._chunked = None

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
            elif restore.chunked:
                restore.orphaned = True
                self.restore_stats.orphaned += 1
                self._abort_chunked(restore, count=False)
            else:
                restore.orphaned = True
                self.restore_stats.orphaned += 1
                del self._restores[request_id]
                # Its load lands later (`_complete` releases its blocks); the
                # room it kept for its request goes now.
                if restore.extra:
                    self.restore_stats.extra_released_unadmitted += restore.extra
                    self._reserved -= restore.extra
                    restore.extra = 0
        self._stop_waiting(request_id)
        self._no_chunk.discard(request_id)
        self._held_now.pop(request_id, None)
        self._alloc_held_since.pop(request_id, None)
        self._last_wait.pop(request_id, None)
        self._admitted_points.pop(request_id, None)
        self._landed_eligible.discard(request_id)
        if self._claim is not None and self._claim[0] == request_id:
            self._claim = None
        arrived = self._arrived.pop(request_id, None)
        if arrived is not None and self.ram_policy is not None:
            self.ram_policy.departed(arrived)
        self._observed.discard(request_id)
        self._admitted_from_ram.discard(request_id)
        parent = self._resumed_from.pop(request_id, None)
        loop_start = self._loop_start(request, parent)
        key = self._remember_end(request, loop_start)
        if key is not None:
            self._keep_resume_points(request, key, parent, loop_start)
        # Held points no session kept now (a rung the ladder did not choose)
        # go like superseded ones: RAM drops its copy, the GPU prunes it.
        self._release_points(self._held_points.pop(request_id, ()))
        self._prune_gpu_states(request)
        if self.ram_policy is not None:
            if key is not None:
                self.ram_policy.add_session(
                    key,
                    _end(request),
                    self._session_units(request, loop_start),
                    parent,
                    rungs=self._rungs_by.get(key),
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
        ladder = self._ladder(request, end)
        # The rungs that are not also near-end points: their states may go on
        # their own (`_may_drop_point`).
        self._rungs_by[key] = {s: p for s, p in ladder.items() if s not in points}
        points.update(dict.fromkeys(ladder))
        # Counted before any are released, so the points they share stay.
        previous = self._kept_by.pop(key, None)
        self._kept_by[key] = (loop_start, tuple(points))
        self._kept.update(list(points))
        if previous is not None:
            self._release_points(previous[1])
        if parent is not None and parent != key:
            # The rungs its parent lost stay lost: its turns resume below.
            lost = self._dropped_rungs.pop(parent, None)
            if lost:
                self._dropped_rungs[key] = [d for d in lost if d[0] < end]
                while len(self._dropped_rungs) > _MAX_DROPPED_SESSIONS:
                    self._dropped_rungs.popitem(last=False)
            self._release_resume_points(parent)
        while len(self._kept_by) > _MAX_SEEN_ENDS:
            self._release_resume_points(next(iter(self._kept_by)))

    def _ladder(self, request: "Request", end: int) -> dict[BlockHash, int]:
        """A finished request's checkpoint ladder: besides its resume points
        near its end, one block-boundary state per doubling stretch of
        distance from its end, [L, 2L), [2L, 4L)... blocks back
        (`ram_ladder_blocks` L), the farthest from the end in each: as the
        session grows, a rung moves to the next stretch and the next one in
        takes its place. RAM evicts a session's tail first, down to its next
        resume point, so with rungs inside the session it loses its tail and
        resumes at a rung (at most 2L blocks back past the first). Without
        them its resume points all sit near its end, and losing those leaves
        no point to resume at. (GPU window 5's nine sessions re-read whole
        were first read that way; at least three of them RAM still held
        whole, and their restores gave up waiting for room: `_wait`.)
        Candidates are the states both tiers keep on its chain (the turns'
        earlier points, other sessions' shared ones) and the rungs it held
        while it computed (`_set_rungs`)."""
        if not self._ladder_blocks or not self.mamba_gids:
            return {}
        hashes, per_block = request.block_hashes, self.units_per_block
        held = set(self._held_points.get(request.request_id, ()))
        end_block = end // self.block_size
        chosen: dict[int, tuple[int, BlockHash]] = {}
        for k in range(1, min(end_block, len(hashes) // per_block) + 1):
            key = hashes[k * per_block - 1]
            if key not in self._kept and key not in held:
                continue
            distance = end_block - k
            if distance < self._ladder_blocks:
                continue
            stretch = (distance // self._ladder_blocks).bit_length()
            # Ascending `k`: the first in its stretch is the farthest back.
            chosen.setdefault(stretch, (k, key))
        return {key: k * self.block_size for k, key in chosen.values()}

    def _hold_prefill_points(
        self, request: "Request", parent: BlockHash | None, extends_end: bool
    ) -> None:
        """Both tiers keep the states at the last block boundaries before an
        admitted request's prompt end and its loop's start from now on, not
        only once it finishes (`_keep_resume_points`): the checkpoint graph
        lets a boundary state go as the request runs past it, and the GPU then
        drops it uncopied. Without them its session resumes only near its end,
        and losing that tail re-reads the session from the start. The ladder's
        rungs the request computes (`_set_rungs`) are held the same way, also
        on a tool loop's turns."""
        hashes, per_block = request.block_hashes, self.units_per_block
        rungs = {
            hashes[b // self.block_size * per_block - 1]: None
            for b in request.retained_boundaries
            if b // self.block_size * per_block <= len(hashes)
        }
        if rungs:
            self._held_points[request.request_id] = tuple(rungs)
            self._kept.update(list(rungs))
            if self.ram_policy is not None:
                self._held_units[request.request_id] = tuple(rungs)
                self.ram_policy.hold(rungs)
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
            self._held_points[request_id] = (
                *self._held_points.get(request_id, ()),
                *points,
            )
            self._kept.update(list(points))
        units.update(points)
        if units and self.ram_policy is not None:
            self._held_units[request_id] = (
                *self._held_units.get(request_id, ()),
                *units,
            )
            self.ram_policy.hold(units)

    def _release_resume_points(self, session: BlockHash) -> None:
        self._rungs_by.pop(session, None)
        kept = self._kept_by.pop(session, None)
        if kept is not None:
            self._release_points(kept[1])

    def _release_points(
        self, states: tuple[BlockHash, ...], only_on_gpu: bool = False
    ) -> None:
        """`only_on_gpu`: prune a released state on the GPU only if the GPU
        holds one of its blocks (a dropped rung's, usually RAM's alone)."""
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
            if self.prune_gpu_states and (not only_on_gpu or self._state_on_gpu(state)):
                self._prune_pending.append(state)

    def _state_on_gpu(self, state: BlockHash) -> bool:
        return any(
            self._on_gpu(make_block_hash_with_group_id(state, group_id))
            for group_id in self.mamba_gids
        )

    def _may_drop_point(self, session: BlockHash, state: BlockHash) -> bool:
        """Whether a ladder point of `session` may go on its own
        (`ram_ladder_ranking`): a rung only it keeps (no running request
        holds it either), no point several continuations share, nobody
        queued resumes at, no spaced state, none loading, and no block of it
        under another key (an exact-end state at a boundary)."""
        if state not in self._rungs_by.get(session, ()):
            return False
        if self._kept[state] != 1 or state in self._queued_states:
            return False
        if self._shared(state):
            return False
        for pool in (self._gpu_block_pool, self.cpu_block_pool):
            if pool is None:
                continue
            cached = pool.cached_block_hash_to_block
            for group_id in self.mamba_gids:
                grouped = make_block_hash_with_group_id(state, group_id)
                for block in cached.get_all_blocks(grouped):
                    num_tokens = block.block_hash_num_tokens
                    if (
                        pool.cached_block_hashes_by_block.get(block.block_id)
                        or (num_tokens is not None and self._spaced(num_tokens))
                        or (
                            pool is self.cpu_block_pool
                            and block.block_id in self._loading_cpu
                        )
                    ):
                        return False
        return True

    def _unkeep_point(self, session: BlockHash, state: BlockHash, why: str) -> None:
        """A ladder point of `session` goes (`why`: "safe", no cut could stop
        there, or "ranked", by value): its states leave RAM (a ranked one's
        are RAM's victims already) and the GPU, and neither keeps it again."""
        rungs = self._rungs_by.get(session)
        if rungs is None or state not in rungs:
            return
        position = rungs.pop(state)
        kept = self._kept_by.get(session)
        if kept is None or state not in kept[1]:
            return
        self._kept_by[session] = (kept[0], tuple(p for p in kept[1] if p != state))
        lost = self._dropped_rungs.setdefault(session, [])
        lost.append((position, why))
        self._dropped_rungs.move_to_end(session)
        while len(self._dropped_rungs) > _MAX_DROPPED_SESSIONS:
            self._dropped_rungs.popitem(last=False)
        self.ladder_events[f"dropped_{why}"] += 1
        self._release_points((state,), only_on_gpu=True)
        logger.log(
            self._restore_log_level,
            "Tiered KV ladder point dropped (%s): session %s at %d tokens",
            why,
            bytes(session)[:6].hex(),
            position,
        )

    def _drain_dropped_points(self) -> None:
        """Drop the ladder points RAM's policy chose since the last step, from
        both tiers, and rank their sessions again."""
        policy = self.ram_policy
        if policy is None:
            return
        dropped = policy.take_dropped_points()
        if not dropped:
            return
        for session, state, why in dropped:
            self._unkeep_point(session, state, why)
        policy.repush(session for session, _, _ in dropped)

    def _note_ladder_outcome(self, session: BlockHash, point: int, seen: int) -> None:
        """A request resumed `session` at `point` tokens: at a rung of its
        ladder (what the rung saved over the one below), or below a rung that
        went, short of the `seen` end it continues (what the drop cost, at
        most: the rung might not have stayed either)."""
        events = self.ladder_events
        positions = self._rungs_by.get(session, {}).values()
        if point in positions:
            events["payoff_resumes"] += 1
            events["payoff_tokens"] += point - max(
                (p for p in positions if p < point), default=0
            )
        lost = [d for d in self._dropped_rungs.get(session, ()) if point < d[0] <= seen]
        if lost:
            position, why = max(lost)
            events[f"regret_{why}_resumes"] += 1
            events[f"regret_{why}_tokens"] += position - point

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
        for restore, pairs in self._unsent:
            # Never sent: nothing reads their blocks.
            restore.reset = True
            self._loading_cpu.difference_update(c.block_id for _, c in pairs)
            self.cpu_block_pool.free_blocks([c for _, c in pairs])
        self._unsent = []
        self._no_chunk.clear()
        self._arrived.clear()
        # The room restores keep goes with them: `_reserved` restarts at 0, so
        # one whose release waits for its load (`_complete`) gives none back.
        for pending in self._restores_by_event.values():
            for restore, _ in pending:
                restore.extra = 0
        for restore in list(self._restores.values()):
            del self._restores[restore.request_id]
            restore.reset = True
            if restore.done or not restore.in_flight:
                self._release(restore)
            elif restore.pending:
                # The chunks in flight land first (`_complete` releases).
                self._loading_cpu.difference_update(b.block_id for b in restore.pending)
                self.cpu_block_pool.free_blocks(restore.pending)
                restore.pending = []
        self._chunked = None
        self.ram_frontiers.clear()
        self._kept.clear()
        self._kept_by.clear()
        self._rungs_by.clear()
        self._dropped_rungs.clear()
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
        self._held_back_since.clear()
        self._wait_credit.clear()
        self._wait_steps.clear()
        self._waiting_need.clear()
        self._held_now.clear()
        self._alloc_held_since.clear()
        self._last_wait.clear()
        self._landed_eligible.clear()
        self._landed_looked_up.clear()
        self._landed_refused.clear()
        self._admitted_points.clear()
        self._claim = None
        self._reserved = 0
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
        # Every admission, a preempted request's again too: where it resumed.
        restored = request_id in self._admitted_from_ram
        self._admitted_from_ram.discard(request_id)
        point = self._admitted_points.pop(request_id, None)
        if point is not None:
            stats = self.restore_stats
            stats.admitted_hit_short += num_computed_tokens < point
            stats.admitted_hit_past += num_computed_tokens > point
        resumes = self.recent_resumes
        resumes[request_id] = (
            "ram" if restored else "gpu" if num_computed_tokens else "cold",
            num_computed_tokens,
        )
        resumes.move_to_end(request_id)
        while len(resumes) > _RECENT_RESUMES:
            resumes.popitem(last=False)
        if request_id in self._observed:
            return
        self._observed.add(request_id)
        seen, key, extends_end = self._seen_end(request)
        self._hold_prefill_points(request, key, extends_end)
        self._note_junction(request, num_computed_tokens, seen)
        if key is None:
            return
        # The session resumed, whichever tier holds it.
        self._resumed_from[request_id] = key
        if extends_end:
            self._extends_end.add(request_id)
        self._note_ladder_outcome(key, num_computed_tokens, seen)
        if self.ram_policy is not None:
            self.ram_policy.resumed(key)
        tier = "ram" if restored else "gpu" if num_computed_tokens else "cold"
        self.resume_stats.record(tier, max(0, seen - num_computed_tokens))
        kind = self._point_kind(request, key, num_computed_tokens)
        self.resume_points[kind, tier] += 1

    def _point_kind(self, request: "Request", session: BlockHash, point: int) -> str:
        """Which of `session`'s points a request resumed at `point` tokens
        landed on (`RESUME_POINT_KINDS`): inside a block, an exact end (any
        token; a prompt tail sits at a hash unit); at a block boundary, a rung of
        its ladder, one of its other kept points (the last boundaries before
        its prompt end, its end, its loop start), a point sessions share, or
        another state the tiers still held."""
        if not point:
            return "none"
        unit = self.hash_block_size
        if point % self.block_size:
            checkpoints = self._checkpoints
            if checkpoints is None or point % unit:
                return "exact_end"
            frontiers = checkpoints.frontiers
            key = frontier_key(frontiers.hash_fn, request, point, unit)
            if key in frontiers.entries or key in self.ram_frontiers.entries:
                return "exact_end"
            return "prompt_tail"
        hashes = request.block_hashes
        if point // unit > len(hashes):
            return "other"
        state = hashes[point // unit - 1]
        if state in self._rungs_by.get(session, ()):
            return "ladder"
        kept = self._kept_by.get(session)
        if kept is not None and state in kept[1]:
            return "near_end"
        if state in self._junctions or self._shared(state):
            return "shared"
        return "other"

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

    def _tier_contents(self) -> None:
        """Count what each tier holds (`tier_blocks`): every block once, by
        `TIER_BLOCK_KINDS`. A free block with a key is cached; one in use
        runs (GPU) or is on its way between the tiers (RAM). RAM's orphans
        are blocks no session owns or will. An attention page is an exact
        end's if a key of it is; a state takes the most telling role of its
        keys: an exact end, a queued request's point, a running request's,
        a ladder rung, a live session's other point, a shared point, a
        prompt tail's (inside a block), or none. Also RAM blocks the GPU
        caches too (`duplicate_blocks`), pages and states."""
        counts = {(t, k): 0 for t in ("gpu", "ram") for k in TIER_BLOCK_KINDS}
        duplicates = {"page": 0, "state": 0}
        gpu = self._gpu_block_pool
        checkpoints = self._checkpoints
        gpu_frontiers = {} if checkpoints is None else checkpoints.frontiers.entries
        held = {s for states in self._held_points.values() for s in states}
        ladder = {s for rungs in self._rungs_by.values() for s in rungs}
        policy, mamba = self.ram_policy, self._mamba_gid_set
        roles = (
            ("queued", self._queued_states),
            ("held", held),
            ("ladder", ladder),
            ("near_end", self._kept),
            ("shared", self._junctions),
        )
        for tier, pool in (("gpu", gpu), ("ram", self.cpu_block_pool)):
            if pool is None:
                continue
            frontiers = gpu_frontiers if tier == "gpu" else self.ram_frontiers.entries
            extra = pool.cached_block_hashes_by_block
            for block in pool.blocks:
                if block.is_null:
                    continue
                if block.ref_cnt:
                    kind = "in_use"
                elif block.block_hash is None:
                    kind = "free"
                elif tier == "ram" and policy is not None and not policy._owned(block):
                    kind = "orphan"
                else:
                    keys = (block.block_hash, *extra.get(block.block_id, ()))
                    units = [get_block_hash(key) for key in keys]
                    exact = any(unit in frontiers for unit in units)
                    group = get_group_id(block.block_hash)
                    num_tokens = block.block_hash_num_tokens
                    if group not in mamba:
                        if group != self.fa_gidx:
                            kind = "other"
                        else:
                            kind = "page_exact_end" if exact else "page"
                    elif exact:
                        kind = "exact_end"
                    else:
                        kind = self._state_role(units, num_tokens, roles)
                counts[tier, kind] += 1
                if (
                    tier == "ram"
                    and block.block_hash is not None
                    and gpu is not None
                    and gpu.cached_block_hash_to_block.get_one_block(block.block_hash)
                    is not None
                ):
                    group = get_group_id(block.block_hash)
                    duplicates["state" if group in mamba else "page"] += 1
        self.tier_blocks, self.duplicate_blocks = counts, duplicates

    def _state_role(
        self,
        units: list[BlockHash],
        num_tokens: int | None,
        roles: tuple[tuple[str, Collection[BlockHash]], ...],
    ) -> str:
        for role, states in roles:
            if any(unit in states for unit in units):
                return role
        if any(map(self._shared, units)):
            return "shared"
        if num_tokens is not None and num_tokens % self.block_size:
            return "prompt_tail"
        return "other"

    def get_stats(self) -> SimpleCPUOffloadStats:
        started = time.perf_counter_ns()
        try:
            return self._get_stats()
        finally:
            self.restore_stats.stats_ns += time.perf_counter_ns() - started

    def _get_stats(self) -> SimpleCPUOffloadStats:
        stats = super().get_stats()
        current, previous = self.restore_stats, self._restore_snapshot
        for metric, fields in (
            (TieredMetricName.RESTORES, RESTORE_EVENT_TO_FIELD),
            (TieredMetricName.ADMISSION, ADMISSION_EVENT_TO_FIELD),
            (TieredMetricName.SEARCH, SEARCH_EVENT_TO_FIELD),
        ):
            for event, field_name in fields.items():
                delta = getattr(current, field_name) - getattr(previous, field_name)
                if delta > 0:
                    stats.increase_counter(metric, delta, (event,))
        for part, field_name in SCHEDULER_PART_TO_FIELD.items():
            delta = getattr(current, field_name) - getattr(previous, field_name)
            if delta > 0:
                stats.increase_counter(
                    TieredMetricName.SCHEDULER_SECONDS, delta / 1e9, (part,)
                )
        pool = self._gpu_block_pool
        if pool is not None:
            spill = (pool.spill_scan_ns, pool.spill_reorder_ns)
            for part, now, then in zip(
                ("spill_scan", "spill_reorder"), spill, self._spill_ns_snapshot
            ):
                if now > then:
                    stats.increase_counter(
                        TieredMetricName.SCHEDULER_SECONDS, (now - then) / 1e9, (part,)
                    )
            self._spill_ns_snapshot = spill
        for kind, field_name in PEAK_KIND_TO_FIELD.items():
            stats.set_gauge(
                TieredMetricName.RESTORE_PEAKS, getattr(current, field_name), (kind,)
            )
        self._restore_snapshot = replace(current)
        for metric, counts, snapshot in (
            (
                TieredMetricName.LOOKUPS,
                self.lookup_outcomes,
                self._lookup_outcomes_snapshot,
            ),
            (
                TieredMetricName.LANDED_WAITS,
                self.landed_waits,
                self._landed_waits_snapshot,
            ),
        ):
            for label, count in counts.items():
                delta = count - snapshot[label]
                if delta > 0:
                    stats.increase_counter(metric, delta, (label,))
        self._lookup_outcomes_snapshot = Counter(self.lookup_outcomes)
        self._landed_waits_snapshot = Counter(self.landed_waits)
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
        for kind, count in self.sessions_lost.items():
            delta = count - self._sessions_lost_snapshot[kind]
            if delta > 0:
                stats.increase_counter(
                    TieredMetricName.RAM_SESSIONS_LOST, delta, (kind,)
                )
        self._sessions_lost_snapshot = Counter(self.sessions_lost)
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
        policy = self.ram_policy
        if policy is not None:
            current, previous = policy.stats, self._ram_evictions_snapshot
            for kind, field_name in (
                ("orphan", "orphans"),
                ("session", "session_blocks"),
                ("rung", "rung_blocks"),
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
            for event, field_name in (
                ("dropped_ranked_blocks", "rung_blocks"),
                ("dropped_fresh_blocks", "rung_fresh"),
            ):
                delta = getattr(current, field_name) - getattr(previous, field_name)
                if delta > 0:
                    stats.increase_counter(TieredMetricName.LADDER, delta, (event,))
            self._ram_evictions_snapshot = replace(current, counts={})
            for labels, count in policy.ladder_p.items():
                delta = count - self._ladder_p_snapshot[labels]
                if delta > 0:
                    stats.increase_counter(TieredMetricName.LADDER_P, delta, labels)
            self._ladder_p_snapshot = Counter(policy.ladder_p)
        for event, count in self.ladder_events.items():
            delta = count - self._ladder_events_snapshot[event]
            if delta > 0:
                stats.increase_counter(TieredMetricName.LADDER, delta, (event,))
        self._ladder_events_snapshot = Counter(self.ladder_events)
        for labels, count in self.resume_points.items():
            delta = count - self._resume_points_snapshot[labels]
            if delta > 0:
                stats.increase_counter(TieredMetricName.RESUMES_BY_POINT, delta, labels)
        self._resume_points_snapshot = Counter(self.resume_points)
        now = time.monotonic()
        if now - self._tier_blocks_at >= _TIER_BLOCKS_EVERY_S:
            self._tier_blocks_at = now
            self._tier_contents()
        if policy is not None and now - self._unranked_at >= _UNRANKED_EVERY_S:
            self._unranked_at = now
            self.unranked_ram_blocks = policy.unranked_blocks()
        for labels, count in self.tier_blocks.items():
            stats.set_gauge(TieredMetricName.TIER_BLOCKS, count, labels)
        for kind, count in self.duplicate_blocks.items():
            stats.set_gauge(TieredMetricName.DUPLICATE_BLOCKS, count, (kind,))
        for kind, count in zip(("page", "state"), self.unranked_ram_blocks):
            stats.set_gauge(TieredMetricName.RAM_UNRANKED_BLOCKS, count, (kind,))
        return stats
