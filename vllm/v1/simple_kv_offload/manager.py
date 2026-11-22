# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for SimpleCPUOffloadConnector."""

import contextlib
import itertools
import time
from collections.abc import Iterable
from dataclasses import dataclass, field, fields, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import (
    MEDIUM_CPU,
    MEDIUM_STORAGE,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    ExternalBlockHash,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
    resolve_block_hashes,
    resolve_dcp_kv_block_size,
    to_event_extra_keys,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
    get_kv_cache_spec_kind,
    get_kv_cache_spec_sliding_window,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.metrics import (
    LAZY_SPILL_EVENT_TO_FIELD,
    OUTCOME_TO_FIELD,
    MetricName,
    SimpleCPUOffloadStats,
)
from vllm.v1.simple_kv_offload.sizing import gpu_total_bytes

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Lazy mode: the most blocks a step stores beyond the walk's target for the
# allocations held for a spill, and how long allocations may be held before
# the spill guard lets them evict blocks the CPU pool does not hold.
_MAX_SHORTFALL_STORE_BLOCKS = 256
_MAX_SPILL_WAIT_S = 10.0


@dataclass
class BlockStoreMeta:
    """Per-block metadata snapshot for BlockStored event emission.

    Captured at store-prep time (when the Request is available) and carried
    through TransferMeta to async completion, where the Request is gone.
    """

    token_ids: list[int]
    parent_block_hash: ExternalBlockHash | None
    lora_id: int | None
    lora_name: str | None
    extra_keys: tuple[Any, ...] | None
    block_size: int | None = None
    lookahead_token_ids: list[int] | None = None


@dataclass
class TransferMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]
    block_meta: list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None = None


@dataclass
class LoadRequestState:
    request: "Request"
    transfer_meta: TransferMeta
    load_event: int | None = None
    finished: bool = False


# NOTE: This per-request state is only used in eager mode.
@dataclass
class StoreRequestState:
    request: "Request"
    # Accumulated block IDs from scheduler_output via yield_req_data.
    block_ids: tuple[list[int], ...]
    # Per-group cursors tracking how many blocks have been stored/skipped.
    num_stored_blocks: list[int]
    store_events: set[int] = field(default_factory=set)
    finished: bool = False


class _StoreAdmission(Enum):
    READY = auto()
    NULL_BLOCK = auto()
    NOT_HASHED = auto()
    IN_FLIGHT = auto()
    ALREADY_CACHED = auto()


@dataclass
class BoundaryStoreStats:
    """Cumulative counters for boundary hand-off stores.

    Every branch that declines a hand-off is silent by design: the boundary is
    simply not offloaded and a later request misses. Without counters the only
    symptom is a lower cache hit rate with nothing in the logs, so each decline
    reason is counted and exposed through
    `SimpleCPUOffloadConnector.get_boundary_store_stats()`. Reset by
    `reset()`, which carries any not-yet-drained deltas; not otherwise
    cleared.
    """

    published: int = 0
    stored: int = 0
    dropped_cpu_full: int = 0
    dropped_request_gone: int = 0
    skipped_already_cached: int = 0
    skipped_in_flight: int = 0
    dropped_null_block: int = 0
    dropped_not_hashed: int = 0


@dataclass
class LazySpillStats:
    """Cumulative counters of lazy mode's spill guard."""

    # GPU blocks stored as they neared eviction.
    stored_blocks: int = 0
    # Allocations (one request, one step) that waited for blocks to be
    # stored first.
    held_allocations: int = 0
    # Blocks the CPU pool could restore, evicted without a copy: the guard
    # had let held allocations go, or no other free block was left.
    unspilled_evicted_blocks: int = 0
    # Times the guard let held allocations go (they waited too long, or the
    # CPU pool had no block to store into).
    guard_releases: int = 0
    # Cached CPU blocks evicted to store new ones.
    cpu_evicted_blocks: int = 0
    # Free blocks the CPU pool could restore that lost their GPU keys
    # without a copy on any path: an allocation, a released or dropped
    # hybrid state unit. Includes unspilled_evicted_blocks.
    uncopied_evicted_blocks: int = 0


class SimpleCPUOffloadScheduler:
    """Scheduler-side manager for CPU offloading."""

    # Lazy mode: a block the GPU drops moves its CPU copy to the CPU pool's
    # most recently used end (`on_evicted`). Off when a CPU eviction policy
    # other than LRU ranks the CPU pool.
    refresh_cpu_on_gpu_eviction = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        scheduler_block_size: int,
        hash_block_size: int,
        lazy_offload: bool = False,
        disk_capacity_bytes: int = 0,
        use_page_cache: bool = False,
        num_cpu_blocks: int | None = None,
        aligned_num_cpu_blocks: int | None = None,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        # When disk mode is active, the offload pool size is disk-based.
        offload_capacity = (
            disk_capacity_bytes if disk_capacity_bytes > 0 else cpu_capacity_bytes
        )
        self.enable_kv_cache_events = (
            vllm_config.kv_events_config is not None
            and vllm_config.kv_events_config.enable_kv_cache_events
        )
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.cp_world_size = dcp_world_size
        self.block_size = scheduler_block_size
        self.hash_block_size = hash_block_size
        assert self.block_size % self.hash_block_size == 0
        # Derive a CPU KVCacheConfig from the GPU config and build a coordinator
        assert kv_cache_config is not None
        self.cpu_kv_cache_config = self._derive_cpu_config(
            kv_cache_config, offload_capacity, num_cpu_blocks, aligned_num_cpu_blocks
        )
        self.num_cpu_blocks = self.cpu_kv_cache_config.num_blocks
        self.prefix_cacheable_group_ids = (
            self.cpu_kv_cache_config.prefix_cacheable_group_ids
        )
        self.kv_event_medium = MEDIUM_STORAGE if disk_capacity_bytes > 0 else MEDIUM_CPU
        # Must track kv_event_medium.
        self._info_labelvalues: tuple[str, ...] = (
            "disk" if disk_capacity_bytes > 0 else "cpu",
            str(use_page_cache).lower(),
            str(lazy_offload).lower(),
            str(self.num_cpu_blocks),
        )
        # Find the full attention kv group for prefix cache matching.
        self.fa_gidx = -1
        for g_idx, g in enumerate(self.cpu_kv_cache_config.kv_cache_groups):
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                self.fa_gidx = g_idx
                break
        assert 0 <= self.fa_gidx < len(self.cpu_kv_cache_config.kv_cache_groups)
        logger.info(
            "SimpleCPUOffloadScheduler: Allocating %d offload blocks "
            "(%.2f GB, mode=%s, backend=%s)",
            self.num_cpu_blocks,
            offload_capacity / (1024**3),
            "lazy" if lazy_offload else "eager",
            "disk" if disk_capacity_bytes > 0 else "cpu",
        )

        spec_config = vllm_config.speculative_config
        num_hash_lookahead_tokens = vllm_config.num_hash_lookahead_tokens
        use_eagle_block_drop = (
            spec_config is not None
            and spec_config.use_eagle_block_drop()
            and not num_hash_lookahead_tokens
        )
        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
            kv_cache_config=self.cpu_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            use_eagle=use_eagle_block_drop,
            enable_caching=True,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=1,
            scheduler_block_size=self.block_size,
            hash_block_size=self.hash_block_size,
            allow_partial_hash_hits=not lazy_offload,
            num_prefill_lookahead=vllm_config.num_prefill_lookahead_tokens,
            num_hash_lookahead_tokens=num_hash_lookahead_tokens,
        )
        self.group_block_sizes = self.cpu_coordinator.group_block_sizes
        # FA group's own resolved block_size; divides scheduler_block_size (the
        # LCM) but is NOT assumed to equal it.
        self.fa_block_size: int = self.group_block_sizes[self.fa_gidx]
        assert self.block_size % self.fa_block_size == 0
        self.cpu_block_pool: BlockPool = self.cpu_coordinator.block_pool
        # GPU block pool reference - bound after scheduler builds kv_cache_manager
        self._gpu_block_pool: BlockPool | None = None

        # Load metadata
        self._reqs_to_load: dict[str, LoadRequestState] = {}
        # Inverse map: load_event_idx -> req_ids. Keyed by load_event_idx because
        # the worker reports completions by event index, not request id.
        self._load_event_to_reqs: dict[int, list[str]] = {}

        # Pending (cpu_hit_blocks, hit_length, num_computed_tokens) tuples from
        # find_longest_cache_hit, kept pinned via touch() while awaiting
        # update_state_after_alloc(). ``num_computed_tokens`` is the local
        # prefix the hit was resolved against; the load path needs it to place
        # external blocks and must not re-derive it from block metadata.
        self._pending_cpu_hits: dict[
            str, tuple[tuple[list[KVCacheBlock], ...], int, int]
        ] = {}
        # Eager mode: local prefix length from get_num_new_matched_tokens(),
        # whose CPU copy update_state_after_alloc() refreshes.
        self._pending_local_hits: dict[str, int] = {}

        # Store metadata
        self._lazy_mode = lazy_offload
        if self._lazy_mode:
            self._target_free = self._estimate_lazy_target_blocks(
                kv_cache_config,
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.cp_world_size,
            )
        else:
            self._target_free = 0
        # Lazy mode: the GPU pool's spill guard (`needs_spill`). Blocks the
        # allocations held last step still need, when they were first held,
        # and whether uncopied blocks are held back now.
        self._spill_shortfall = 0
        self._held_since: float | None = None
        self._holding = True
        # Boundary siblings (`_boundary_siblings`) copied without a reference,
        # so they keep their place in the eviction order: GPU block id -> its
        # key when the copy started. The guard holds them back like any
        # uncopied block; a block that changed hands mid-copy backs no CPU key.
        # Once copied, a sibling still leaves only in eviction order
        # (`keeps_order`).
        self._unpinned_stores: dict[int, BlockHashWithGroupId] = {}
        self._early_pages: dict[int, BlockHashWithGroupId] = {}
        self.lazy_spill_stats = LazySpillStats()
        self._lazy_spill_snapshot = LazySpillStats()
        self._store_event_to_blocks: dict[int, TransferMeta] = {}
        self._abandoned_store_event_to_blocks: dict[int, TransferMeta] = {}
        # Eager mode only
        self._reqs_to_store: dict[str, StoreRequestState] = {}
        self._store_event_to_reqs: dict[int, list[str]] = {}
        self._in_flight_store_gpu_blocks: set[int] = set()
        self._pending_finished_stores: list[TransferMeta] = []
        self._abandoned_reqs_to_load: dict[str, LoadRequestState] = {}

        # Event counters
        self._load_event_counter: int = 0
        self._store_event_counter: int = 0
        self.boundary_store_stats = BoundaryStoreStats()
        # Interval stats state drained by get_stats()
        self._boundary_stats_snapshot = BoundaryStoreStats()
        self._interval_load_blocks_completed = 0

        # For TP/PP: track partial store completions across steps.
        # Events must be reported by all world_size workers before considered complete.
        self._expected_worker_count = vllm_config.parallel_config.world_size
        self._store_event_pending_counts: dict[int, int] = {}

    @staticmethod
    def _derive_cpu_config(
        gpu_config: "KVCacheConfig",
        cpu_capacity_bytes: int,
        num_cpu_blocks: int | None = None,
        aligned_num_cpu_blocks: int | None = None,
    ) -> "KVCacheConfig":
        """Derive a CPU KVCacheConfig from the GPU config.
        Same kv_cache_groups, num_blocks scaled by CPU/GPU memory ratio.

        An explicit `num_cpu_blocks` is checked against local capacity.
        `aligned_num_cpu_blocks` comes from the live worker allocations and
        is used as-is when no explicit count is configured."""
        # Import here to avoid potential circular imports
        from vllm.v1.kv_cache_interface import KVCacheTensor

        assert len(gpu_config.kv_cache_tensors) > 0

        num_gpu_blocks = gpu_config.num_blocks
        local_capacity = (
            num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes(gpu_config)
        )
        if num_cpu_blocks is None:
            num_cpu_blocks = (
                aligned_num_cpu_blocks
                if aligned_num_cpu_blocks is not None
                else max(1, local_capacity)
            )
        elif (
            type(num_cpu_blocks) is not int or not 0 < num_cpu_blocks <= local_capacity
        ):
            raise ValueError(
                f"num_cpu_blocks={num_cpu_blocks!r} must be between 1 and "
                f"local offload capacity {local_capacity}"
            )
        # Create CPU kv_cache_tensors mirroring GPU by scaling size proportionally.
        cpu_tensors = [
            KVCacheTensor(
                size=t.size // num_gpu_blocks * num_cpu_blocks,
                layers=list(t.layers),
                layer_stride=t.layer_stride,
                block_stride=t.block_stride,
                offset=t.offset,
            )
            for t in gpu_config.kv_cache_tensors
        ]

        return replace(
            gpu_config,
            num_blocks=num_cpu_blocks,
            kv_cache_tensors=cpu_tensors,
        )

    @staticmethod
    def _estimate_lazy_target_blocks(
        kv_cache_config: "KVCacheConfig",
        max_num_batched_tokens: int,
        cp_world_size: int = 1,
    ) -> int:
        """GPU blocks to keep available (free/offloaded) per step in lazy mode."""
        WATERMARK_RATIO = 1.0  # Reserve larger space to avoid running out of GPU blocks
        target = 0
        for g in kv_cache_config.kv_cache_groups:
            spec = g.kv_cache_spec
            block_size = resolve_dcp_kv_block_size(spec, cp_world_size)
            if not spec.prefix_cacheable:
                # Bounded-replay SWA (DSV4.1) is never offloaded, but its blocks for
                # each scheduled token come from the same free queue.
                if isinstance(spec, SlidingWindowSpec):
                    target += cdiv(max_num_batched_tokens, block_size)
            elif isinstance(spec, MambaSpec):
                target += 2
            elif isinstance(spec, SlidingWindowSpec):
                target += cdiv(spec.sliding_window, block_size) + 1
            else:
                target += cdiv(max_num_batched_tokens, block_size)
        return int(target * (1 + WATERMARK_RATIO))

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        """Bind GPU block pool so that we can touch blocks during stores.
        Called by Scheduler after kv_cache_manager is ready."""
        self._gpu_block_pool = gpu_block_pool
        gpu_block_pool.lower_tier = self
        if self._lazy_mode:
            # Lazy stores extend the GPU pool instead of mirroring it: a block
            # the CPU pool could restore leaves the GPU only once stored.
            gpu_block_pool.spill_guard = self

    def restores(self, block: "KVCacheBlock") -> bool:
        """LowerKVTier: whether the CPU pool can load the cached GPU `block`
        back once it is evicted. Lazy mode stores every such block before it
        leaves (the spill guard); eager mode holds what it stored."""
        if block.block_hash is None or block.is_null or not self._restorable(block):
            return False
        if self._lazy_mode:
            return True
        cached = self.cpu_block_pool.cached_block_hash_to_block
        return cached.get_one_block(block.block_hash) is not None

    @property
    def holding(self) -> bool:
        """SpillGuard: whether uncopied blocks are held back now."""
        return self._holding

    def needs_spill(self, block: "KVCacheBlock") -> bool:
        """SpillGuard: whether evicting the free cached GPU `block` now would
        lose what a CPU lookup could restore."""
        return self._uncopied(block)

    def _uncopied(self, block: "KVCacheBlock") -> bool:
        """Whether the free cached GPU `block` is one a CPU lookup could
        restore and the CPU pool does not hold yet. A page no GPU state needs
        any more still counts: it is the prefix of states the CPU pool
        holds."""
        if (
            block.block_hash is None
            or block.is_null
            or block.ref_cnt
            or not self._restorable(block)
        ):
            return False
        cached = self.cpu_block_pool.cached_block_hash_to_block
        return cached.get_one_block(block.block_hash) is None

    def keeps_order(self, block: "KVCacheBlock") -> bool:
        """SpillGuard: whether `block` is a boundary sibling copied ahead of
        its turn. Leaving ahead of the uncopied blocks before it would split
        its session around a block neither tier serves from."""
        return (
            block.block_hash is not None
            and self._early_pages.get(block.block_id) == block.block_hash
        )

    def on_allocation_held(self, num_blocks: int) -> None:
        """SpillGuard: the next walk also stores `num_blocks` blocks."""
        self._spill_shortfall += num_blocks
        self.lazy_spill_stats.held_allocations += 1

    def has_pending_block_frees(self) -> bool:
        """Whether stores in flight, or the next walk, will let go blocks an
        allocation was held for: the scheduler then waits a step instead of
        preempting a request."""
        return self._lazy_mode and self._holding and self._spill_shortfall > 0

    def on_unspilled_eviction(self, blocks: list["KVCacheBlock"]) -> None:
        """SpillGuard: `blocks` leave the GPU without a copy."""
        self.lazy_spill_stats.unspilled_evicted_blocks += len(blocks)
        logger.warning_once(
            "SimpleCPU lazy offload evicted GPU blocks before storing them "
            "(vllm:simple_kv_offload_lazy_spill_total, "
            "event=unspilled_evicted_blocks)."
        )

    def on_evicted(self, block: "KVCacheBlock") -> None:
        """SpillGuard: the GPU drops `block`, so its CPU copy is now the only
        one: it moves to the CPU pool's most recently used end, with the
        copies of its boundary siblings (the CPU serves the boundary only
        whole). The CPU's LRU order is otherwise its store order, and a block
        stored long ago but hit on the GPU since (lazy mode never refreshes
        the CPU on GPU hits) would be the next the CPU evicts, losing a
        session from both tiers."""
        assert block.block_hash is not None
        self._early_pages.pop(block.block_id, None)
        cpu = self.cpu_block_pool.cached_block_hash_to_block
        if (
            cpu.get_one_block(block.block_hash) is None
            and not block.ref_cnt
            and not block.is_null
            and self._restorable(block)
        ):
            self.lazy_spill_stats.uncopied_evicted_blocks += 1
        if not self.refresh_cpu_on_gpu_eviction:
            return
        block_hash = get_block_hash(block.block_hash)
        copies = []
        for group_id in self.prefix_cacheable_group_ids:
            cpu_block = cpu.get_one_block(
                make_block_hash_with_group_id(block_hash, group_id)
            )
            if cpu_block is not None and cpu_block.ref_cnt == 0:
                copies.append(cpu_block)
        if copies:
            self.cpu_block_pool.touch(copies)
            self.cpu_block_pool.free_blocks(copies)

    def _update_holding(self, num_cpu_free: int) -> None:
        """Hold back uncopied blocks unless allocations have waited for stores
        too long or the CPU pool has no block to store into."""
        if not self._spill_shortfall:
            self._held_since = None
            self._holding = True
            return
        now = time.monotonic()
        if self._held_since is None:
            self._held_since = now
        holding = num_cpu_free > 0 and now - self._held_since < _MAX_SPILL_WAIT_S
        if self._holding and not holding:
            self.lazy_spill_stats.guard_releases += 1
            self._held_since = None
        self._holding = holding

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Return (num_new_tokens, is_async) from consecutive CPU cache hits."""
        # Pins found CPU blocks so they survive LRU eviction until
        # update_state_after_alloc() consumes them. Any pin from an earlier
        # call on the same request (e.g. retry after a failed allocate_slots)
        # is dropped first.
        if stale := self._pending_cpu_hits.pop(request.request_id, None):
            self._free_pending_cpu_hit(stale)
        self._pending_local_hits.pop(request.request_id, None)

        if request.skip_reading_external_kv:
            return 0, False

        if num_computed_tokens and not self._lazy_mode:
            # Refreshed at admission: a request waiting for blocks repeats
            # this lookup every step.
            self._pending_local_hits[request.request_id] = num_computed_tokens

        if num_computed_tokens % self.block_size != 0:
            # Transfers are whole-block copies, so an external suffix cannot
            # start in the middle of a destination block. When the GPU-side
            # coordinator also serves fine-grained hits, the local prefix can
            # land on a hash boundary that is not a scheduler-block boundary;
            # such a request keeps its local hit and skips the external one.
            logger.warning_once(
                "SimpleCPU external lookup requires scheduler-block-aligned "
                "local tokens, got %d tokens with scheduler block size %d.",
                num_computed_tokens,
                self.block_size,
            )
            return 0, False

        # Suffix-relative chunk boundaries must match absolute token positions.
        for group_id in self.prefix_cacheable_group_ids:
            spec = self.cpu_kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
            if (
                isinstance(spec, ChunkedLocalAttentionSpec)
                and num_computed_tokens % spec.attention_chunk_size
            ):
                return 0, False

        num_skipped_hashes = num_computed_tokens // self.hash_block_size
        remaining_hashes = request.block_hashes[num_skipped_hashes:]

        if not remaining_hashes:
            return 0, False
        # Must recompute at least the last token, matching the logic in
        # kv_cache_manager.get_computed_blocks().
        max_hit_len = request.num_tokens - 1 - num_computed_tokens
        if max_hit_len <= 0:
            return 0, False
        cpu_hit_blocks, hit_length, _ = self.cpu_coordinator.find_longest_cache_hit(
            remaining_hashes, max_hit_len
        )

        if hit_length > 0:
            pin_blocks = [
                blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null
            ]
            self.cpu_block_pool.touch(pin_blocks)
            self._pending_cpu_hits[request.request_id] = (
                cpu_hit_blocks,
                hit_length,
                num_computed_tokens,
            )
            return hit_length, True
        return 0, False

    # TODO(yifan): this API now only matches the suffix part of the prefix cache. A more
    # general API should scan blocks in both GPU and CPU block pool in a single pass.
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id
        block_ids_by_group = blocks.get_block_ids()
        num_groups = len(block_ids_by_group)

        # Store tracking (eager mode only). Register the request;
        # block IDs are accumulated from scheduler_output in
        # _prepare_eager_store_specs via yield_req_data.
        if not self._lazy_mode and req_id not in self._reqs_to_store:
            self._reqs_to_store[req_id] = StoreRequestState(
                request=request,
                block_ids=tuple([] for _ in range(num_groups)),
                num_stored_blocks=[0] * num_groups,
            )

        if num_local_tokens := self._pending_local_hits.pop(req_id, None):
            # Eager CPU is inclusive: a GPU prefix hit also uses the CPU copy,
            # but eager stores skip it, so move it to the MRU end here.
            prefix_hit, _, _ = self.cpu_coordinator.find_longest_cache_hit(
                request.block_hashes, num_local_tokens
            )
            prefix_blocks = [b for grp in prefix_hit for b in grp if not b.is_null]
            self.cpu_block_pool.touch(prefix_blocks)
            self.cpu_block_pool.free_blocks(reversed(prefix_blocks))

        # Pop the CPU hit cached by get_num_new_matched_tokens(). The
        # found blocks were pinned there to survive LRU eviction in the window
        # between get_num_new_matched_tokens() and this matching call.
        pending = self._pending_cpu_hits.pop(req_id, None)

        if num_external_tokens == 0:
            if pending is not None:
                logger.warning(
                    "SimpleCPUOffloadScheduler: update_state_after_alloc "
                    "called for req_id=%s with no external tokens but "
                    "get_num_new_matched_tokens() unexpectedly recorded "
                    "a pending CPU hit; releasing the stale pin.",
                    req_id,
                )
                self._free_pending_cpu_hit(pending)
            return

        if pending is None:
            logger.warning(
                "SimpleCPUOffloadScheduler: update_state_after_alloc called "
                "for req_id=%s with num_external_tokens=%d but no pending "
                "CPU hit from get_num_new_matched_tokens(); skipping load.",
                req_id,
                num_external_tokens,
            )
            return

        cpu_hit_blocks_full, _, num_computed_tokens = pending

        assert num_computed_tokens % self.block_size == 0, (
            "SimpleCPU external loads must start at a scheduler-block boundary"
        )
        # Fine-grained hits are hash-block aligned, not group-block aligned, so
        # the per-group block count below rounds up instead of dividing exactly.
        # Hash alignment is still required: it is what makes the accepted
        # external suffix start on a boundary every group can address.
        assert num_external_tokens % self.hash_block_size == 0, (
            f"num_external_tokens={num_external_tokens} is not aligned to "
            f"hash_block_size={self.hash_block_size}"
        )

        # The scheduler may have accepted fewer blocks than
        # get_num_new_matched_tokens() reported.
        # (e.g. due to token budget in test_partial_gpu_prefix_plus_cpu_load).
        # Take only the leading N blocks per group matching num_external_tokens;
        # the rest will be released along with the temp pin below.
        cpu_hit_blocks: list[list[KVCacheBlock]] = []
        for g in range(num_groups):
            if g not in self.prefix_cacheable_group_ids:
                cpu_hit_blocks.append([])
                continue
            g_block_size = self.group_block_sizes[g]
            n_take_g = cdiv(num_external_tokens, g_block_size)
            cpu_hit_blocks.append(cpu_hit_blocks_full[g][:n_take_g])

        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        cpu_blocks_to_touch: list[KVCacheBlock] = []

        for g in range(num_groups):
            cpu_blocks_g = cpu_hit_blocks[g]
            n_ext_g = len(cpu_blocks_g)
            if n_ext_g == 0:
                continue

            g_block_size = self.group_block_sizes[g]
            gpu_ext_start = num_computed_tokens // g_block_size
            group_gpu_ids = block_ids_by_group[g]

            for i, cpu_blk in enumerate(cpu_blocks_g):
                # Skip null blocks (e.g. sliding window or mamba padding).
                if cpu_blk.is_null:
                    continue
                gpu_block_ids.append(group_gpu_ids[gpu_ext_start + i])
                cpu_block_ids.append(cpu_blk.block_id)
                cpu_blocks_to_touch.append(cpu_blk)

        # Touch CPU blocks to prevent eviction during async load.
        self.cpu_block_pool.touch(cpu_blocks_to_touch)
        # Release the temporary pin held since get_num_new_matched_tokens().
        self._free_pending_cpu_hit(pending)

        # Touch GPU blocks to prevent freeing during async load
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.touch(
            [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
        )

        assert self._reqs_to_load.get(req_id) is None
        self._reqs_to_load[req_id] = LoadRequestState(
            request=request, transfer_meta=TransferMeta(gpu_block_ids, cpu_block_ids)
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimpleCPUOffloadMetadata:
        # --- Stores ---
        store_event = -1
        store_gpu, store_cpu, store_req_ids, store_meta = self.prepare_store_specs(
            scheduler_output
        )
        for transfer in self._pending_finished_stores:
            store_gpu.extend(transfer.gpu_block_ids)
            store_cpu.extend(transfer.cpu_block_ids)
            if store_meta is not None:
                assert transfer.block_meta is not None
                store_meta.extend(transfer.block_meta)
        self._pending_finished_stores.clear()

        if store_gpu:
            store_event = self._store_event_counter
            self._store_event_counter += 1
            self._store_event_to_blocks[store_event] = TransferMeta(
                store_gpu, store_cpu, store_meta
            )
            if store_req_ids:  # For eager mode only, track req->blocks mapping
                self._store_event_to_reqs[store_event] = store_req_ids
                for req_id in store_req_ids:
                    store_state = self._reqs_to_store.get(req_id)
                    if store_state is not None:
                        store_state.store_events.add(store_event)

        # --- Loads ---
        load_event = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        load_req_ids: list[str] = []
        for req_id, load_state in self._reqs_to_load.items():
            if load_state.load_event is not None:
                continue
            assert load_state.transfer_meta is not None
            load_gpu.extend(load_state.transfer_meta.gpu_block_ids)
            load_cpu.extend(load_state.transfer_meta.cpu_block_ids)
            load_req_ids.append(req_id)
        if load_req_ids:
            load_event = self._load_event_counter
            self._load_event_counter += 1
            for req_id in load_req_ids:
                self._reqs_to_load[req_id].load_event = load_event
            self._load_event_to_reqs[load_event] = load_req_ids

        result = SimpleCPUOffloadMetadata(
            load_event=load_event,
            load_gpu_blocks=load_gpu,
            load_cpu_blocks=load_cpu,
            load_event_to_reqs={
                event_idx: list(req_ids)
                for event_idx, req_ids in self._load_event_to_reqs.items()
            },
            store_event=store_event,
            store_gpu_blocks=store_gpu,
            store_cpu_blocks=store_cpu,
            need_flush=bool(scheduler_output.preempted_req_ids),
        )
        return result

    def prepare_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[
        list[int],
        list[int],
        list[str],
        list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None,
    ]:
        """Prepare store specs for the store event."""
        if self._lazy_mode:
            return self._prepare_lazy_store_specs()
        else:
            return self._prepare_eager_store_specs(scheduler_output)

    def _prepare_lazy_store_specs(
        self,
    ) -> tuple[
        list[int],
        list[int],
        list[str],
        list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None,
    ]:
        """Offload cached GPU blocks near eviction.

        Walks the GPU pool's eviction order (its free queue's, or its eviction
        policy's) and stores the blocks the spill guard holds back
        (`needs_spill`). Stops once target_free blocks are covered (free or
        stored) and as many blocks are stored as the allocations held in the
        last step still need (up to a per-step cap), or at CPU capacity.
        """
        gpu_pool = self._gpu_block_pool
        if gpu_pool is None or self._target_free <= 0:
            return [], [], [], None

        cpu_pool = self.cpu_block_pool
        num_cpu_free = cpu_pool.get_num_free_blocks()
        shortfall = min(self._spill_shortfall, _MAX_SHORTFALL_STORE_BLOCKS)
        self._update_holding(num_cpu_free)
        self._spill_shortfall = 0

        policy = gpu_pool.eviction_policy
        order: Iterable[KVCacheBlock] = (
            policy.peek_eviction_order(2 * (self._target_free + shortfall) + 16)
            if policy is not None
            else gpu_pool.free_block_queue.iter_blocks_after(None)
        )
        gpu_ids: list[int] = []
        pages: list[int] = []
        chosen: set[int] = set()
        for covered, node in enumerate(order):
            if len(chosen) >= num_cpu_free or (
                covered >= self._target_free and len(chosen) >= shortfall
            ):
                break
            if (
                node.block_id in chosen
                or node.block_id in self._unpinned_stores
                or not self.needs_spill(node)
            ):
                continue
            # A block the CPU pool holds may still be held back for its
            # siblings (a unit that leaves the GPU whole).
            own = [node] if self._uncopied(node) else []
            siblings = [
                page
                for page in self._boundary_siblings(node)
                if page.block_id not in chosen
            ]
            # The unit is copied whole or not at all, within the CPU pool's
            # free blocks: its siblings can take it past them.
            if len(chosen) + len(own) + len(siblings) > num_cpu_free:
                break
            for block in own:
                gpu_ids.append(block.block_id)
                chosen.add(block.block_id)
            for page in siblings:
                assert page.block_hash is not None
                pages.append(page.block_id)
                chosen.add(page.block_id)
                self._unpinned_stores[page.block_id] = page.block_hash

        # Batch-allocate CPU blocks.
        if chosen:
            self.lazy_spill_stats.stored_blocks += len(chosen)
            self.lazy_spill_stats.cpu_evicted_blocks += sum(
                1
                for block in itertools.islice(
                    cpu_pool.free_block_queue.iter_blocks_after(None), len(chosen)
                )
                if block.block_hash is not None
            )
            cpu_blocks = cpu_pool.get_new_blocks(len(chosen))
            cpu_ids = [blk.block_id for blk in cpu_blocks]
            # Touch GPU blocks to prevent eviction during async copy.
            gpu_pool.touch([gpu_pool.blocks[bid] for bid in gpu_ids])
        else:
            cpu_ids = []

        return gpu_ids + pages, cpu_ids, [], None

    def describe_residency(self, request: "Request") -> str | None:
        """Debugging: per group, where each of the request's block hashes is
        cached: B in both tiers, G the GPU only, C the CPU only, . neither
        (lower case: the GPU block is in use). None when nothing is."""
        assert self._gpu_block_pool is not None
        gpu = self._gpu_block_pool.cached_block_hash_to_block
        cpu = self.cpu_block_pool.cached_block_hash_to_block
        lines = []
        cached = False
        for group_id in range(len(self.group_block_sizes)):
            marks = []
            for block_hash in request.block_hashes:
                key = make_block_hash_with_group_id(block_hash, group_id)
                on_gpu = gpu.get_one_block(key)
                in_cpu = cpu.get_one_block(key) is not None
                mark = "BG"[not in_cpu] if on_gpu is not None else "C."[not in_cpu]
                if on_gpu is not None and on_gpu.ref_cnt:
                    mark = mark.lower()
                cached |= mark != "."
                marks.append(mark)
            lines.append(f"g{group_id} {''.join(marks)}")
        return " | ".join(lines) if cached else None

    def _boundary_siblings(self, block: "KVCacheBlock") -> list["KVCacheBlock"]:
        """The GPU blocks of the other groups keyed by `block`'s hash (the
        full-attention page ending where a state sits, the other Mamba groups'
        states there) that the CPU pool does not hold yet. A CPU lookup serves
        a boundary only with all of them, while the eviction order can take
        one group's block long before another's: the scheduler then combines
        the GPU's full-attention pages with the CPU's states
        (`supports_divergent_local_hybrid_hits`), which needs the CPU pool to
        hold the boundary whole. A full-attention page may still be in use (a
        running request freed a state as it moved on): a cached page is full
        and no longer written. A state in use may not be written yet."""
        assert block.block_hash is not None and self._gpu_block_pool is not None
        group_id = get_group_id(block.block_hash)
        if self.group_block_sizes[group_id] != self.fa_block_size:
            return []
        gpu = self._gpu_block_pool.cached_block_hash_to_block
        cpu = self.cpu_block_pool.cached_block_hash_to_block
        block_hash = get_block_hash(block.block_hash)
        siblings = []
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
                or sibling.block_id in self._unpinned_stores
                or (sibling.ref_cnt and sibling_group != self.fa_gidx)
                or not self._restorable(sibling)
                or cpu.get_one_block(key) is not None
            ):
                continue
            siblings.append(sibling)
        return siblings

    def _restorable(self, block: "KVCacheBlock") -> bool:
        """Whether a lazy CPU lookup can serve this cached block: it walks
        whole blocks only, so a key that ends inside its group's block (a
        fine-grained hit's partial page, or a hybrid frontier checkpoint)
        would only take CPU space."""
        assert block.block_hash is not None
        num_tokens = block.block_hash_num_tokens
        group_size = self.group_block_sizes[get_group_id(block.block_hash)]
        return num_tokens is not None and num_tokens % group_size == 0

    def _prepare_eager_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[
        list[int],
        list[int],
        list[str],
        list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None,
    ]:
        """Identify newly computed blocks to offload from scheduler requests.

        Only considers blocks whose KV data has been **confirmed computed** by
        the GPU. Blocks from the current step are stored on a later step, or by
        the finish-time flush if the request completes first.

        Returns:
            (gpu_block_ids, cpu_block_ids, req_ids, block_meta) for the store
            event. `block_meta` is None when kv cache events are disabled.

        """
        merged_gpu_block_ids: list[int] = []
        merged_cpu_block_ids: list[int] = []
        req_ids: list[str] = []
        merged_block_meta: list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None = (
            [] if self.enable_kv_cache_events else None
        )

        gpu_block_pool = self._gpu_block_pool
        if gpu_block_pool is None:
            return [], [], [], merged_block_meta
        cpu_block_pool = self.cpu_block_pool
        num_groups = len(self.cpu_kv_cache_config.kv_cache_groups)
        # Dedup against blocks already scheduled.
        in_flight = self._in_flight_store_gpu_blocks
        num_free = cpu_block_pool.get_num_free_blocks()

        block_state = scheduler_output.kv_connector_block_state
        boundary_offloads = (
            block_state.boundary_state_offloads if block_state is not None else {}
        )
        stats = self.boundary_store_stats
        preempted_req_ids = scheduler_output.preempted_req_ids or ()
        for req_id, entries in boundary_offloads.items():
            # The scheduler drains handoffs without filtering by request
            # liveness. A request that finished or was preempted in this step
            # is giving up its blocks, which may already have been reallocated
            # to another request; reading them would publish unrelated KV under
            # a valid hash. Drop conservatively, as the mooncake store does.
            store_state = self._reqs_to_store.get(req_id)
            if (
                store_state is None
                or store_state.finished
                or req_id in preempted_req_ids
                or req_id in scheduler_output.finished_req_ids
            ):
                stats.published += len(entries)
                stats.dropped_request_gone += len(entries)
                continue
            scheduled_for_req = False
            # ``boundary_tokens`` is not needed here: the cache key comes from
            # the handed-off block itself, not from the offered boundary.
            for _, gpu_block_id, _ in entries:
                stats.published += 1
                gpu_block = gpu_block_pool.blocks[gpu_block_id]
                admission = self._classify_store_candidate(gpu_block)
                if admission is _StoreAdmission.NULL_BLOCK:
                    stats.dropped_null_block += 1
                    continue
                if admission is _StoreAdmission.NOT_HASHED:
                    stats.dropped_not_hashed += 1
                    continue
                if admission is _StoreAdmission.IN_FLIGHT:
                    stats.skipped_in_flight += 1
                    continue
                if admission is _StoreAdmission.ALREADY_CACHED:
                    stats.skipped_already_cached += 1
                    continue
                if num_free <= 0:
                    stats.dropped_cpu_full += 1
                    logger.warning_once(
                        "SimpleCPU dropped a boundary-state handoff because "
                        "the CPU block pool is full; this boundary will not be retried."
                    )
                    continue
                cpu_block = cpu_block_pool.get_new_blocks(1)[0]
                merged_gpu_block_ids.append(gpu_block_id)
                merged_cpu_block_ids.append(cpu_block.block_id)
                if merged_block_meta is not None:
                    # Keep the metadata list index-parallel with the block ids.
                    # The hand-off block's own hashes are resolved at completion,
                    # so there is nothing to capture here; emit the event without
                    # a payload, as the size-mismatch fallback already does.
                    merged_block_meta.append({})
                in_flight.add(gpu_block_id)
                gpu_block_pool.touch([gpu_block])
                num_free -= 1
                stats.stored += 1
                scheduled_for_req = True
            if scheduled_for_req:
                req_ids.append(req_id)

        # Preemption frees the request's blocks. It resumes with a full new
        # block table in its NewRequestData.
        for req_id in preempted_req_ids:
            state = self._reqs_to_store.get(req_id)
            if state is not None:
                state.block_ids = tuple([] for _ in range(num_groups))
                state.num_stored_blocks = [0] * num_groups

        for req_id, block_id_groups, is_full_table in yield_req_data(scheduler_output):
            state = self._reqs_to_store.get(req_id)
            if state is None or state.finished:
                continue

            if is_full_table:
                state.block_ids = tuple([] for _ in range(num_groups))

            if block_id_groups:
                for g in range(min(num_groups, len(block_id_groups))):
                    if block_id_groups[g] is not None:
                        # Accumulate new block IDs.
                        state.block_ids[g].extend(block_id_groups[g])

            num_new_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_new_tokens == 0:
                continue

            block_ids_by_group = state.block_ids
            if not block_ids_by_group:
                continue

            gpu_block_ids, advanced_per_group, block_meta = (
                self._select_eager_blocks_to_store(state, block_ids_by_group)
            )

            # Batch allocate the CPU destinations.
            n_to_alloc = len(gpu_block_ids)
            if n_to_alloc > 0:
                cpu_blocks_alloc = cpu_block_pool.get_new_blocks(n_to_alloc)
                cpu_block_ids = [blk.block_id for blk in cpu_blocks_alloc]
            else:
                cpu_block_ids = []

            if cpu_block_ids:
                req_ids.append(req_id)
                merged_gpu_block_ids.extend(gpu_block_ids)
                merged_cpu_block_ids.extend(cpu_block_ids)
                in_flight.update(gpu_block_ids)
                if merged_block_meta is not None:
                    assert block_meta is not None
                    merged_block_meta.extend(block_meta)

                # Touch GPU blocks to prevent freeing during async copy
                gpu_block_pool.touch(
                    [gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
                )

                logger.debug(
                    "Request %s: Scheduling store of %d blocks to CPU (%d groups)",
                    req_id,
                    len(cpu_block_ids),
                    num_groups,
                )

            # Advance per-group cursors (includes cached hits + newly stored)
            for g in range(num_groups):
                state.num_stored_blocks[g] += advanced_per_group[g]

        # A request can contribute both boundary and positional blocks to the
        # same event. Completion tracking is set-based, so keep one request ID.
        req_ids = list(dict.fromkeys(req_ids))
        return merged_gpu_block_ids, merged_cpu_block_ids, req_ids, merged_block_meta

    def _cached_gpu_block(
        self, resolved_hashes: "BlockHashList", block_idx: int, group_id: int
    ) -> "KVCacheBlock | None":
        """Return the GPU block still cached under this group's block hash."""
        if block_idx >= len(resolved_hashes):
            return None
        assert self._gpu_block_pool is not None
        blocks = self._gpu_block_pool.get_cached_block(
            resolved_hashes[block_idx], [group_id]
        )
        return blocks[0] if blocks else None

    def _select_eager_blocks_to_store(
        self,
        state: StoreRequestState,
        block_ids_by_group: tuple[list[int], ...],
    ) -> tuple[
        list[int],
        list[int],
        list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None,
    ]:
        """Return confirmed eager blocks, cursor advances, and event metadata."""
        assert self._gpu_block_pool is not None
        gpu_block_ids: list[int] = []
        advanced_per_group = [0] * len(self.cpu_kv_cache_config.kv_cache_groups)
        block_meta: list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None = (
            [] if self.enable_kv_cache_events else None
        )
        request = state.request
        confirmed_tokens = request.num_computed_tokens - request.num_output_placeholders
        num_free = self.cpu_block_pool.get_num_free_blocks()

        for g, group_gpu_ids in enumerate(block_ids_by_group):
            if g not in self.prefix_cacheable_group_ids:
                continue
            if len(gpu_block_ids) >= num_free:
                break
            group_manager = self.cpu_coordinator.single_type_managers[g]
            if not group_manager.has_positionally_stable_blocks:
                continue
            # FIXME (yifan): handle CPU cache eviction, where
            # num_stored_blocks can be stale and omit evicted blocks in
            # the middle of the request.
            group_size = self.group_block_sizes[g]
            # Mimic the GPU KV cache manager's cacheable prefix.
            cacheable_tokens = self.cpu_coordinator.get_num_cacheable_tokens(
                confirmed_tokens, g
            )
            cacheable_tokens = min(
                cacheable_tokens, len(request.block_hashes) * self.hash_block_size
            )
            ready = min(len(group_gpu_ids), cacheable_tokens // group_size)
            resolved_hashes = resolve_block_hashes(
                request.block_hashes, self.hash_block_size, group_size
            )
            curr_mm_idx = 0
            secondary_mm_idx = 0
            start = state.num_stored_blocks[g]
            for i, gpu_block_id in enumerate(group_gpu_ids[start:ready], start=start):
                gpu_block = self._gpu_block_pool.blocks[gpu_block_id]
                if gpu_block.is_null:
                    # Sliding-window groups null pages that left the window
                    # before the connector sees the block table, but the
                    # retained prefix-cache tail stays hashed in the GPU free
                    # queue; store it from there.
                    recovered = self._cached_gpu_block(resolved_hashes, i, g)
                    if recovered is None:
                        advanced_per_group[g] += 1
                        continue
                    gpu_block = recovered
                    gpu_block_id = gpu_block.block_id
                if (
                    self._classify_store_candidate(gpu_block)
                    is not _StoreAdmission.READY
                ):
                    advanced_per_group[g] += 1
                    continue
                if len(gpu_block_ids) >= num_free:
                    break
                primary_block_hash = gpu_block.block_hash
                assert primary_block_hash is not None
                gpu_block_ids.append(gpu_block_id)
                advanced_per_group[g] += 1
                if block_meta is not None:
                    token_start = i * group_size
                    token_end = token_start + group_size
                    parent_hash = (
                        None
                        if i == 0
                        else maybe_convert_block_hash(resolved_hashes[i - 1])
                    )
                    lora_req = request.lora_request
                    extra_keys, curr_mm_idx = (
                        self.cpu_block_pool.generate_event_extra_keys(
                            request, token_start, token_end, curr_mm_idx
                        )
                    )
                    meta_by_hash = {
                        primary_block_hash: BlockStoreMeta(
                            token_ids=list(
                                request.all_token_ids[token_start:token_end]
                            ),
                            parent_block_hash=parent_hash,
                            lora_id=lora_req.adapter_id if lora_req else None,
                            lora_name=lora_req.name if lora_req else None,
                            extra_keys=extra_keys,
                            lookahead_token_ids=(
                                self.cpu_block_pool.get_lookahead_token_ids(
                                    request, token_end
                                )
                                if self.cpu_block_pool.num_hash_lookahead_tokens
                                else None
                            ),
                        )
                    }
                    first_hash_idx = token_start // self.hash_block_size
                    last_hash_idx = token_end // self.hash_block_size
                    for hash_idx in range(first_hash_idx, last_hash_idx):
                        block_hash = make_block_hash_with_group_id(
                            request.block_hashes[hash_idx], g
                        )
                        if block_hash is None or block_hash == primary_block_hash:
                            continue
                        hash_start = hash_idx * self.hash_block_size
                        hash_end = hash_start + self.hash_block_size
                        secondary_extra_keys, secondary_mm_idx = (
                            self.cpu_block_pool.generate_event_extra_keys(
                                request,
                                hash_start,
                                hash_end,
                                secondary_mm_idx,
                            )
                        )
                        meta_by_hash[block_hash] = BlockStoreMeta(
                            token_ids=request.all_token_ids[hash_start:hash_end],
                            parent_block_hash=(
                                None
                                if hash_idx == 0
                                else maybe_convert_block_hash(
                                    request.block_hashes[hash_idx - 1]
                                )
                            ),
                            lora_id=lora_req.adapter_id if lora_req else None,
                            lora_name=lora_req.name if lora_req else None,
                            extra_keys=secondary_extra_keys,
                            block_size=self.hash_block_size,
                            lookahead_token_ids=(
                                self.cpu_block_pool.get_lookahead_token_ids(
                                    request, hash_end
                                )
                                if self.cpu_block_pool.num_hash_lookahead_tokens
                                else None
                            ),
                        )
                    block_meta.append(meta_by_hash)

        return gpu_block_ids, advanced_per_group, block_meta

    def _classify_store_candidate(self, gpu_block: "KVCacheBlock") -> _StoreAdmission:
        if gpu_block.is_null:
            return _StoreAdmission.NULL_BLOCK
        if gpu_block.block_hash is None:
            return _StoreAdmission.NOT_HASHED
        if gpu_block.block_id in self._in_flight_store_gpu_blocks:
            return _StoreAdmission.IN_FLIGHT
        if (
            self.cpu_block_pool.cached_block_hash_to_block.get_one_block(
                gpu_block.block_hash
            )
            is not None
        ):
            return _StoreAdmission.ALREADY_CACHED
        return _StoreAdmission.READY

    def get_boundary_store_stats(self) -> BoundaryStoreStats:
        return replace(self.boundary_store_stats)

    def get_stats(self) -> SimpleCPUOffloadStats:
        """Drain per-step stats for the connector's stats hooks."""
        stats = SimpleCPUOffloadStats()

        current = self.boundary_store_stats
        for outcome, field_name in OUTCOME_TO_FIELD.items():
            delta = getattr(current, field_name) - getattr(
                self._boundary_stats_snapshot, field_name
            )
            if delta > 0:
                stats.increase_counter(MetricName.SAVE_OUTCOMES, delta, (outcome,))
        self._boundary_stats_snapshot = replace(current)

        spill = self.lazy_spill_stats
        for event, field_name in LAZY_SPILL_EVENT_TO_FIELD.items():
            delta = getattr(spill, field_name) - getattr(
                self._lazy_spill_snapshot, field_name
            )
            if delta > 0:
                stats.increase_counter(MetricName.LAZY_SPILL, delta, (event,))
        self._lazy_spill_snapshot = replace(spill)

        if self._interval_load_blocks_completed:
            stats.increase_counter(
                MetricName.LOAD_BLOCKS,
                self._interval_load_blocks_completed,
            )
            self._interval_load_blocks_completed = 0

        stats.set_gauge(
            MetricName.USED_BLOCKS,
            self.num_cpu_blocks - self.cpu_block_pool.get_num_free_blocks(),
        )
        pending = sum(
            len(t.cpu_block_ids)
            for t in (
                *self._store_event_to_blocks.values(),
                *self._pending_finished_stores,
                *self._abandoned_store_event_to_blocks.values(),
            )
        )
        stats.set_gauge(MetricName.PENDING_STORE_BLOCKS, pending)
        stats.set_gauge(MetricName.INFO, 1, self._info_labelvalues)
        return stats

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Handle async transfer completions from worker.

        Load completions arrive via finished_recving (real req_ids).
        Store completions arrive via kv_connector_worker_meta as
        per-event worker counts. We accumulate across steps and process
        a store event only when all workers have reported completion.
        """
        # --- Load completions ---
        for req_id in list(connector_output.finished_recving or []):
            completed_blocks = self._cleanup_load_request(req_id)
            if completed_blocks:
                self._interval_load_blocks_completed += completed_blocks

        # --- Store completions ---
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, SimpleCPUOffloadWorkerMetadata):
            return
        for event_idx, count in meta.completed_store_events.items():
            total = self._store_event_pending_counts.get(event_idx, 0) + count
            if total >= self._expected_worker_count:
                self._store_event_pending_counts.pop(event_idx, None)
                self._process_store_event(event_idx)
            else:
                self._store_event_pending_counts[event_idx] = total

    def _process_store_event(self, event_idx: int) -> None:
        """Process a fully-completed store event."""
        transfer = self._store_event_to_blocks.pop(event_idx, None)
        if transfer is None:
            transfer = self._abandoned_store_event_to_blocks.pop(event_idx, None)
            if transfer is None:
                return  # guard stale events from before a reset() call
            self._release_transfer_refs(transfer)
            return

        if not self._lazy_mode:
            self._in_flight_store_gpu_blocks.difference_update(transfer.gpu_block_ids)

        self._process_store_completion(
            transfer.gpu_block_ids,
            transfer.cpu_block_ids,
            transfer.block_meta,
        )
        logger.debug(
            "Store event %d completed: cached %d blocks to CPU",
            event_idx,
            len(transfer.cpu_block_ids),
        )

        # Eager only: update per-req state
        if not self._lazy_mode:
            for req_id in self._store_event_to_reqs.pop(event_idx, []):
                state = self._reqs_to_store.get(req_id)
                if state is None:
                    continue
                state.store_events.discard(event_idx)
                if state.finished and not state.store_events:
                    self._cleanup_store_request(req_id)

    def _process_store_completion(
        self,
        gpu_block_ids: list[int],
        cpu_block_ids: list[int],
        block_meta: list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None = None,
    ) -> None:
        """Register copied blocks in the CPU prefix cache and release refs."""
        assert len(cpu_block_ids) == len(gpu_block_ids)
        assert block_meta is None or len(block_meta) == len(gpu_block_ids)

        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in cpu_block_ids]

        assert self._gpu_block_pool is not None
        stored: list[BlockHashWithGroupId] = []
        pinned: list[int] = []
        for i, (gpu_block_id, cpu_block) in enumerate(zip(gpu_block_ids, cpu_blocks)):
            gpu_block = self._gpu_block_pool.blocks[gpu_block_id]
            primary_hash = gpu_block.block_hash
            unpinned_key = self._unpinned_stores.pop(gpu_block_id, None)
            if unpinned_key is None:
                pinned.append(gpu_block_id)
            elif primary_hash != unpinned_key:
                # Evicted (and maybe reused) while copied without a reference.
                continue
            else:
                self._early_pages[gpu_block_id] = unpinned_key
            if primary_hash is None:
                # Hybrid state checkpoints drop a unit's keys from all of its
                # blocks when one of them is evicted, also from a block being
                # copied. Its copy then backs no key: free it uncached.
                continue
            stored.append(primary_hash)
            self.cpu_block_pool._insert_block_hash(
                primary_hash,
                cpu_block,
                num_tokens=gpu_block.block_hash_num_tokens,
            )
            secondary_hashes = self._gpu_block_pool.cached_block_hashes_by_block.get(
                gpu_block_id, ()
            )
            for block_hash in secondary_hashes:
                self.cpu_block_pool._insert_block_hash(
                    block_hash, cpu_block, num_tokens=None
                )
            if self.enable_kv_cache_events:
                meta_by_hash = block_meta[i] if block_meta is not None else {}
                primary_group_idx = get_group_id(primary_hash)
                primary_block_size = self.group_block_sizes[primary_group_idx]
                events_to_emit: list[
                    tuple[BlockHashWithGroupId, BlockStoreMeta | None, int]
                ] = [
                    (
                        primary_hash,
                        meta_by_hash.get(primary_hash),
                        primary_block_size,
                    )
                ]
                for secondary_hash in secondary_hashes:
                    secondary_meta = meta_by_hash.get(secondary_hash)
                    events_to_emit.append(
                        (
                            secondary_hash,
                            secondary_meta,
                            secondary_meta.block_size
                            if secondary_meta is not None
                            and secondary_meta.block_size is not None
                            else 0,
                        )
                    )

                for block_hash, meta, event_block_size in events_to_emit:
                    group_idx = get_group_id(block_hash)
                    spec = self.cpu_kv_cache_config.kv_cache_groups[
                        group_idx
                    ].kv_cache_spec
                    if meta is not None and len(meta.token_ids) != event_block_size:
                        # token_ids were sliced with a different g_block_size
                        # (e.g. Mamba+DCP where capture uses spec.block_size *
                        # cp_world_size but the event emits spec.block_size * 1).
                        # Emit empty metadata until #49962 fixes the sizing.
                        token_ids = []
                        parent_block_hash = None
                        extra_keys = None
                    else:
                        token_ids = meta.token_ids if meta else []
                        parent_block_hash = meta.parent_block_hash if meta else None
                        extra_keys = meta.extra_keys if meta else None
                    self.cpu_block_pool.kv_event_queue.append(
                        BlockStored(
                            block_hashes=[
                                maybe_convert_block_hash(get_block_hash(block_hash))
                            ],
                            parent_block_hash=parent_block_hash,
                            token_ids=token_ids,
                            block_size=event_block_size,
                            lora_id=meta.lora_id if meta else None,
                            medium=self.kv_event_medium,
                            lora_name=meta.lora_name if meta else None,
                            extra_keys=to_event_extra_keys(extra_keys and [extra_keys]),
                            group_idx=group_idx,
                            kv_cache_spec_kind=get_kv_cache_spec_kind(spec).value,
                            kv_cache_spec_sliding_window=(
                                get_kv_cache_spec_sliding_window(spec)
                            ),
                            locality="LOCAL",
                            lookahead_token_ids=(
                                [meta.lookahead_token_ids]
                                if token_ids
                                and meta is not None
                                and meta.lookahead_token_ids is not None
                                else None
                            ),
                        )
                    )

        policy = self._gpu_block_pool.eviction_policy
        if policy is not None and stored and not self._lazy_mode:
            # Lazy mode counts every restorable block as stored already.
            policy.lower_tier_stored(stored)

        # Free CPU and GPU blocks' ref counts to turn them into prefix cache.
        # Lazy stores come from the GPU eviction end; return them there.
        self.cpu_block_pool.free_blocks(cpu_blocks)
        self._gpu_block_pool.free_blocks(
            (self._gpu_block_pool.blocks[bid] for bid in pinned),
            evict_first=self._lazy_mode,
        )

    def _release_transfer_refs(self, transfer: TransferMeta) -> None:
        """Release transfer refs without making copied data cacheable."""
        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in transfer.cpu_block_ids]
        self.cpu_block_pool.free_blocks(cpu_blocks)
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.free_blocks(
            self._gpu_block_pool.blocks[bid]
            for bid in transfer.gpu_block_ids
            if self._unpinned_stores.pop(bid, None) is None
        )

    def has_pending_stores(self) -> bool:
        """Return True if a store transfer is queued or in flight."""
        return bool(
            self._pending_finished_stores
            or self._store_event_to_blocks
            or self._abandoned_store_event_to_blocks
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Always returns (False, None). GPU blocks are protected by ref_cnt,
        so the scheduler can free blocks immediately."""
        req_id = request.request_id

        # Release any temp CPU hit pin from get_num_new_matched_tokens()
        # if request is canceled or preempted before update_state_after_alloc()
        pending = self._pending_cpu_hits.pop(req_id, None)
        if pending is not None:
            self._free_pending_cpu_hit(pending)
        self._pending_local_hits.pop(req_id, None)

        # Handle load: defer cleanup if load is in-flight
        load_state = self._reqs_to_load.get(req_id)
        if load_state is not None:
            if load_state.load_event is not None:
                load_state.finished = True  # Defer: load in-flight
            else:
                self._cleanup_load_request(req_id)

        # Handle store (eager mode only): defer cleanup if stores in-flight
        if not self._lazy_mode:
            store_state = self._reqs_to_store.get(req_id)
            if store_state is not None:
                if store_state.store_events:
                    store_state.finished = True  # Defer: stores in-flight
                else:
                    self._cleanup_store_request(req_id)

        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._queue_finished_eager_store(request, block_ids)
        return self.request_finished(request, block_ids=[])

    def _queue_finished_eager_store(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> None:
        """Queue confirmed eager blocks omitted after a request enters decode."""
        state = self._reqs_to_store.get(request.request_id)
        gpu_pool = self._gpu_block_pool
        if state is None or gpu_pool is None:
            return
        gpu_ids, _, block_meta = self._select_eager_blocks_to_store(state, block_ids)
        partial_tail = self._find_fa_partial_tail_source(request, block_ids)
        if (
            partial_tail is not None
            # Decode tokens can fill the boundary block, in which case the
            # positional scan above already selected it. Appending again would
            # allocate two CPU blocks for one GPU block.
            and partial_tail not in gpu_ids
            and len(gpu_ids) < self.cpu_block_pool.get_num_free_blocks()
        ):
            gpu_ids.append(partial_tail)
            if block_meta is not None:
                # The block's own hashes are registered at completion, so no
                # capture is needed here; emit the event without a payload, as
                # the size-mismatch fallback above already does.
                block_meta.append({})
        if not gpu_ids:
            return
        cpu_blocks = self.cpu_block_pool.get_new_blocks(len(gpu_ids))
        self._pending_finished_stores.append(
            TransferMeta(gpu_ids, [b.block_id for b in cpu_blocks], block_meta)
        )
        self._in_flight_store_gpu_blocks.update(gpu_ids)
        gpu_pool.touch([gpu_pool.blocks[bid] for bid in gpu_ids])

    def _find_fa_partial_tail_source(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> int | None:
        """Locate the full-attention prompt-tail block for a fine-grained hit.

        Mirrors `FullAttentionManager._cache_partial_tail_block`: only the
        final prompt hash boundary is eligible, and boundaries that land on a
        physical block edge are already covered by the positional scan.

        Attention block tables are append-only, so the block is located
        positionally, as the mooncake store and offloading connector also do;
        no new core hand-off is required. The boundary key is registered at
        completion along with the block's other hashes, so nothing has to be
        captured here.
        """
        if not self.cpu_coordinator.enable_partial_hash_hits:
            return None
        assert self._gpu_block_pool is not None
        manager = self.cpu_coordinator.single_type_managers[self.fa_gidx]
        token_limit = (
            request.num_prompt_tokens
            if manager.use_eagle
            else request.num_prompt_tokens - 1
        )
        boundary_tokens = (
            min(token_limit // self.hash_block_size, len(request.block_hashes))
            * self.hash_block_size
        )
        if boundary_tokens == 0 or boundary_tokens > request.num_computed_tokens:
            return None
        if boundary_tokens % self.fa_block_size == 0:
            return None
        block_idx = boundary_tokens // self.fa_block_size
        fa_block_ids = block_ids[self.fa_gidx]
        if block_idx >= len(fa_block_ids):
            return None
        gpu_block_id = fa_block_ids[block_idx]
        gpu_block = self._gpu_block_pool.blocks[gpu_block_id]
        if gpu_block.is_null or gpu_block_id in self._in_flight_store_gpu_blocks:
            return None
        hash_idx = boundary_tokens // self.hash_block_size - 1
        if hash_idx >= len(request.block_hashes):
            return None
        block_hash = make_block_hash_with_group_id(
            request.block_hashes[hash_idx], self.fa_gidx
        )
        # Only worth copying when the boundary key is actually registered on
        # the source block; completion replays the block's own hashes.
        if (
            gpu_block.block_hash != block_hash
            and block_hash
            not in self._gpu_block_pool.cached_block_hashes_by_block.get(
                gpu_block_id, ()
            )
        ):
            return None
        if self.cpu_block_pool.cached_block_hash_to_block.get_one_block(block_hash):
            return None
        return gpu_block_id

    def _free_pending_cpu_hit(
        self, pending: tuple[tuple[list["KVCacheBlock"], ...], int, int]
    ) -> None:
        """Release the temporary CPU block pin taken in get_num_new_matched_tokens()."""
        cpu_hit_blocks, _hit_length, _num_computed_tokens = pending
        blocks_to_free = [
            blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null
        ]
        if blocks_to_free:
            self.cpu_block_pool.free_blocks(blocks_to_free)

    def _cleanup_load_request(self, req_id: str) -> int:
        """Release all load resources for a request.

        Shared between request_finished() and update_connector_output() paths.
        Removes the request from _reqs_to_load, cleans up event mappings,
        and frees CPU/GPU touch refs.

        Returns the number of blocks in the load if it had been issued to
        the worker, else 0 (never-issued loads are not counted as completed
        by the caller).
        """
        state = self._reqs_to_load.pop(req_id, None)
        if state is None:
            state = self._abandoned_reqs_to_load.pop(req_id, None)
        if state is None:
            return 0
        # Remove from load event mapping (only this req, not whole event)
        if state.load_event is not None:
            reqs = self._load_event_to_reqs.get(state.load_event)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._load_event_to_reqs.pop(state.load_event, None)

        if state.transfer_meta is not None:
            # Free CPU touch refs
            self.cpu_block_pool.free_blocks(
                self.cpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.cpu_block_ids
            )
            # Free GPU touch refs
            assert self._gpu_block_pool is not None
            self._gpu_block_pool.free_blocks(
                self._gpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.gpu_block_ids
            )

        if state.load_event is None:
            return 0
        return len(state.transfer_meta.gpu_block_ids)

    def _cleanup_store_request(self, req_id: str) -> None:
        """Release store metadata for a request.

        Metadata-only cleanup but no block freeing. Job completion handles
        block caching and GPU ref freeing via _process_store_completion().
        """
        state = self._reqs_to_store.pop(req_id, None)
        if state is None:
            return
        for event_idx in list(state.store_events):
            if (reqs := self._store_event_to_reqs.get(event_idx)) is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._store_event_to_reqs.pop(event_idx, None)
        state.store_events.clear()

    def take_events(self) -> Iterable[KVCacheEvent]:
        events = self.cpu_block_pool.take_events()
        for event in events:
            if isinstance(event, BlockRemoved):
                event.medium = self.kv_event_medium
                event.locality = "LOCAL"
        return events

    def reset(self) -> bool:
        """Abandon pending transfers and reset the CPU cache when safe.

        Worker-side DMA may still be using blocks after reset is requested.
        Keep those block refs pinned until the existing completion path reports
        the transfer finished, then release refs without caching abandoned
        store results.
        """
        for pending in self._pending_cpu_hits.values():
            self._free_pending_cpu_hit(pending)
        self._pending_cpu_hits.clear()
        self._pending_local_hits.clear()

        self._abandoned_store_event_to_blocks.update(self._store_event_to_blocks)
        for transfer in self._pending_finished_stores:
            self._release_transfer_refs(transfer)
        self._pending_finished_stores.clear()

        self._store_event_to_blocks.clear()
        self._in_flight_store_gpu_blocks.clear()

        # Loads that have not been sent to the worker cannot have running DMA.
        # In-flight loads stay pinned and are cleaned up on completion.
        for req_id in list(self._reqs_to_load):
            state = self._reqs_to_load.pop(req_id)
            if state.load_event is None:
                self._reqs_to_load[req_id] = state
                self._cleanup_load_request(req_id)
            else:
                self._abandoned_reqs_to_load[req_id] = state

        self._reqs_to_store.clear()
        self._store_event_to_reqs.clear()
        self._store_event_pending_counts = {
            event_idx: count
            for event_idx, count in self._store_event_pending_counts.items()
            if event_idx in self._abandoned_store_event_to_blocks
        }
        # Seed the fresh counters with outcomes since the last drain so a
        # mid-interval reset() does not drop them from the next report.
        undrained = BoundaryStoreStats(
            **{
                f.name: max(
                    0,
                    getattr(self.boundary_store_stats, f.name)
                    - getattr(self._boundary_stats_snapshot, f.name),
                )
                for f in fields(BoundaryStoreStats)
            }
        )
        self.boundary_store_stats = undrained
        self._boundary_stats_snapshot = BoundaryStoreStats()
        self._spill_shortfall = 0
        self._held_since = None
        self._holding = True
        # NOTE: _load_event_counter / _store_event_counter are not
        # reset as they are monotonic and must stay ahead of the workers
        # high-water marks to avoid event index collisions

        if self._abandoned_store_event_to_blocks or self._abandoned_reqs_to_load:
            return False

        return self.cpu_block_pool.reset_prefix_cache()
