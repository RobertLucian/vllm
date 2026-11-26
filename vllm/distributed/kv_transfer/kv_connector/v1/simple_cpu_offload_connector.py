# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SimpleCPUOffloadConnector: minimal CPU KV cache offloading."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.host_buffer import parse_hugepage_size
from vllm.v1.simple_kv_offload.manager import (
    BoundaryStoreStats,
    SimpleCPUOffloadScheduler,
)
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.v1.simple_kv_offload.metrics import (
    SimpleCPUOffloadPromMetrics,
    SimpleCPUOffloadStats,
)
from vllm.v1.simple_kv_offload.worker import (
    SimpleCPUOffloadWorker,
)

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Default CPU capacity: 8 GB
DEFAULT_CPU_CAPACITY_BYTES = 8 * (1024**3)

VALID_KV_OFFLOAD_BACKENDS = ("cpu", "disk")
# Lazy mode: how a cached GPU block the CPU pool lacks gets its copy. "walk":
# ahead of its eviction, which waits for the copy's acknowledgement. "inline":
# in the step that reuses it, ordered on the device before that step's writes.
VALID_SPILL_MODES = ("walk", "inline")
# Keys that only apply to the disk backend, warned about under "cpu".
_DISK_ONLY_KEYS = (
    "disk_path",
    "disk_capacity_bytes",
    "disk_buffer_slots",
    "use_page_cache",
)


def _find_simple_cpu_offload_connector(
    connector: KVConnectorBase_V1,
) -> "SimpleCPUOffloadConnector | None":
    if isinstance(connector, SimpleCPUOffloadConnector):
        return connector
    child_connectors = getattr(connector, "_connectors", None)
    if child_connectors:
        for child in child_connectors:
            found = _find_simple_cpu_offload_connector(child)
            if found is not None:
                return found
    return None


class SimpleCPUOffloadConnector(KVConnectorBase_V1, SupportsHMA):
    """CPU KV cache offloading with custom kernel transfers and BlockPool LRU."""

    # A subclass's scheduler-side manager, and whether it always runs lazy.
    scheduler_manager_cls: type[SimpleCPUOffloadScheduler] = SimpleCPUOffloadScheduler
    always_lazy = False

    @classmethod
    def supports_lookahead_block_hashes(
        cls, kv_transfer_config: "KVTransferConfig"
    ) -> bool:
        return True

    @classmethod
    def supports_spec_frontier_checkpoints(
        cls, kv_transfer_config: "KVTransferConfig"
    ) -> bool:
        # No store reads a Mamba state before its gather or copy has run:
        # - Eager stores skip align-mode Mamba groups (not positionally stable)
        #   and drop a finishing request's boundary-state hand-offs.
        # - The lazy walk takes only free blocks. A gather or copy holds its
        #   blocks until the step that runs it is scheduled, and past that
        #   step's completion when frees are deferred.
        # - Workers launch stores after the step's forward, ordered behind the
        #   gathers and copies on the compute stream.
        # - Inline spills (`spill="inline"`) take only free blocks too, with
        #   frees always deferred (`spills_at_reuse`), and workers launch them
        #   behind the step's gathers.
        return True

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        # Lazy stores leave a session split across tiers: the GPU evicts its
        # tail first, so its first full-attention pages outlive the recurrent
        # states, which only the CPU pool still holds. A local hit at the
        # full-attention depth plus a CPU hit past it restores the session.
        return self._lazy_offload

    @property
    def requires_kv_delivery(self) -> bool:
        # Runs as kv_both, but is a best-effort cache: a dropped save is just a
        # future cache miss, so opt out of the producer-role default.
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        enable_prefix_caching = vllm_config.cache_config.enable_prefix_caching
        extra_config = self._kv_transfer_config.kv_connector_extra_config or {}

        cpu_capacity_bytes = int(
            extra_config.get("cpu_bytes_to_use", DEFAULT_CPU_CAPACITY_BYTES)
        )
        # cpu_bytes_to_use is server-wide for compatibility;
        # cpu_bytes_to_use_per_rank overrides for per-rank capacity.
        world_size = vllm_config.parallel_config.world_size
        cpu_capacity_per_rank = cpu_capacity_bytes // world_size
        if "cpu_bytes_to_use_per_rank" in extra_config:
            explicit = int(extra_config["cpu_bytes_to_use_per_rank"])
            if explicit != cpu_capacity_per_rank:
                logger.warning(
                    "cpu_bytes_to_use_per_rank (%.2f GB) != "
                    "cpu_bytes_to_use/world_size (%.2f GB). Using per-rank value.",
                    explicit / (1024**3),
                    cpu_capacity_per_rank / (1024**3),
                )
            cpu_capacity_per_rank = explicit

        # When set, num_cpu_blocks overrides automatic pool sizing from byte budgets
        # (including cpu_bytes_to_use and cpu_bytes_to_use_per_rank), allowing ranks
        # with different KV layouts (e.g. PP stages) to use the same pool size.
        # Byte budgets still bound capacity.
        num_cpu_blocks = extra_config.get("num_cpu_blocks")
        if num_cpu_blocks is not None and (
            type(num_cpu_blocks) is not int or num_cpu_blocks <= 0
        ):
            raise ValueError("num_cpu_blocks must be a positive integer")

        lazy_offload = self.always_lazy or bool(extra_config.get("lazy_offload", False))
        self._lazy_offload = lazy_offload

        kv_offload_backend = str(extra_config.get("kv_offload_backend", "cpu"))
        if kv_offload_backend not in VALID_KV_OFFLOAD_BACKENDS:
            raise ValueError(
                f"Unknown kv_offload_backend {kv_offload_backend!r}; "
                f"expected one of {VALID_KV_OFFLOAD_BACKENDS}"
            )
        disk_mode = kv_offload_backend == "disk"
        self._cache_hit_source = (
            CacheHitSource.DISK if disk_mode else CacheHitSource.HOST
        )

        spill = str(extra_config.get("spill", "walk"))
        if spill not in VALID_SPILL_MODES:
            raise ValueError(
                f"Unknown spill {spill!r}; expected one of {VALID_SPILL_MODES}"
            )
        if spill == "inline" and (not lazy_offload or disk_mode):
            raise ValueError(
                'spill="inline" requires lazy_offload and kv_offload_backend="cpu".'
            )
        # Checked here: the GPU pool gets its eviction policy only after the
        # connector is bound to it.
        if spill == "inline" and vllm_config.cache_config.prefix_cache_eviction != (
            "lru"
        ):
            raise ValueError(
                'spill="inline" copies GPU blocks as they leave the free queue, '
                "in its LRU order; use --prefix-cache-eviction lru."
            )

        disk_path = extra_config.get("disk_path", None) or None
        disk_capacity_bytes = int(
            extra_config.get("disk_capacity_bytes", 100 * (1024**3))
        )
        disk_buffer_slots = max(1, int(extra_config.get("disk_buffer_slots", 2)))
        use_page_cache = bool(extra_config.get("use_page_cache", False))
        cpu_hugepage_size = parse_hugepage_size(
            extra_config.get("cpu_hugepage_block_size")
        )

        if disk_mode:
            if disk_path is None:
                raise ValueError(
                    'kv_offload_backend="disk" requires disk_path to be set.'
                )
            if disk_capacity_bytes <= 0:
                raise ValueError(
                    'kv_offload_backend="disk" requires disk_capacity_bytes > 0.'
                )
            if cpu_hugepage_size is not None:
                raise ValueError(
                    'cpu_hugepage_block_size applies only to kv_offload_backend="cpu".'
                )
        else:
            ignored = [k for k in _DISK_ONLY_KEYS if k in extra_config]
            if ignored:
                logger.warning(
                    'kv_offload_backend is "cpu", ignoring disk-only config: %s. '
                    'Set kv_offload_backend="disk" to enable the disk backend.',
                    ", ".join(ignored),
                )
            disk_path = None

        self.scheduler_manager: SimpleCPUOffloadScheduler | None = None
        self.worker_handler: SimpleCPUOffloadWorker | None = None

        if not enable_prefix_caching:
            logger.warning(
                "Detected prefix caching disabled, disabling CPU offload "
                "since it requires prefix caching."
            )
            return

        logger.info(
            "SimpleCPUOffloadConnector: role=%s, "
            "per_rank=%.2f GB, world_size=%d, mode=%s, backend=%s, disk=%s",
            role.name,
            cpu_capacity_per_rank / (1024**3),
            world_size,
            "lazy" if lazy_offload else "eager",
            kv_offload_backend,
            disk_path or "none",
        )
        if spill == "inline":
            logger.info(
                "SimpleCPUOffloadConnector: lazy stores spill inline, in the "
                "step that reuses a block."
            )
        if num_cpu_blocks is not None:
            logger.info(
                "Using explicit num_cpu_blocks=%d for offload pool sizing "
                "instead of automatic sizing from byte budgets.",
                num_cpu_blocks,
            )

        if role == KVConnectorRole.SCHEDULER:
            from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

            assert kv_cache_config is not None
            scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
                kv_cache_config, vllm_config
            )
            aligned_num_cpu_blocks = (
                vllm_config.cache_config.simple_cpu_offload_num_blocks
            )
            self.scheduler_manager = self.scheduler_manager_cls(
                vllm_config,
                kv_cache_config,
                cpu_capacity_per_rank,
                scheduler_block_size=scheduler_block_size,
                hash_block_size=hash_block_size,
                lazy_offload=lazy_offload,
                disk_capacity_bytes=disk_capacity_bytes if disk_mode else 0,
                use_page_cache=use_page_cache if disk_mode else False,
                num_cpu_blocks=num_cpu_blocks,
                aligned_num_cpu_blocks=aligned_num_cpu_blocks,
                spill=spill,
                **self._scheduler_manager_kwargs(extra_config),
            )
        elif role == KVConnectorRole.WORKER:
            self.worker_handler = SimpleCPUOffloadWorker(
                vllm_config,
                kv_cache_config,
                cpu_capacity_per_rank,
                kv_offload_backend=kv_offload_backend,
                disk_path=disk_path,
                disk_capacity_bytes=disk_capacity_bytes,
                disk_buffer_slots=disk_buffer_slots,
                use_page_cache=use_page_cache,
                num_cpu_blocks=num_cpu_blocks,
                cpu_hugepage_size=cpu_hugepage_size,
                spill=spill,
            )

    def _scheduler_manager_kwargs(self, extra_config: dict[str, Any]) -> dict[str, Any]:
        """Extra constructor arguments of `scheduler_manager_cls`."""
        return {}

    # --- Worker-side methods ---

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self.worker_handler is not None:
            self.worker_handler.register_kv_caches(kv_caches)

    def get_aligned_num_cpu_blocks(self) -> int | None:
        if self.worker_handler is None:
            return None
        num_cpu_blocks = self.worker_handler.num_cpu_blocks
        return num_cpu_blocks if num_cpu_blocks > 0 else None

    def bind_connector_metadata(
        self,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        super().bind_connector_metadata(connector_metadata)
        if self.worker_handler is not None:
            assert isinstance(connector_metadata, SimpleCPUOffloadMetadata)
            self.worker_handler.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self.worker_handler is not None:
            self.worker_handler.clear_connector_metadata()

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        if self.worker_handler is not None:
            assert isinstance(kv_connector_metadata, SimpleCPUOffloadMetadata)
            self.worker_handler.handle_preemptions(kv_connector_metadata)

    def spill_before_reuse(self, connector_metadata: KVConnectorMetadata) -> None:
        if self.worker_handler is not None:
            assert isinstance(connector_metadata, SimpleCPUOffloadMetadata)
            self.worker_handler.spill_before_reuse(connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if self.worker_handler is not None:
            self.worker_handler.start_load_kv()

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass  # Always load asynchronously, issued in start_load_kv()

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        pass  # Always save asynchronously, issued in wait_for_save()

    def wait_for_save(self) -> None:
        if self.worker_handler is not None:
            self.worker_handler.wait_for_save()

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        if self.worker_handler is not None:
            return self.worker_handler.get_finished(finished_req_ids)
        return None, None

    def build_connector_worker_meta(self):
        if self.worker_handler is not None:
            return self.worker_handler.build_connector_worker_meta()
        return None

    def shutdown(self) -> None:
        if self.worker_handler is not None:
            self.worker_handler.shutdown()

    # --- Scheduler-side methods ---

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.bind_gpu_block_pool(gpu_block_pool)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.get_num_new_matched_tokens(
                request, num_computed_tokens
            )
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_state_after_alloc(
                request, blocks, num_external_tokens
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.build_connector_meta(scheduler_output)
        return SimpleCPUOffloadMetadata()

    def has_pending_block_frees(self) -> bool:
        # Lazy mode: an allocation held for stores goes ahead next step.
        return (
            self.scheduler_manager is not None
            and self.scheduler_manager.has_pending_block_frees()
        )

    @property
    def spills_at_reuse(self) -> bool:
        # Lazy mode's inline spills read a free block in the step that takes
        # it: no step in flight may still write it.
        return (
            self.scheduler_manager is not None
            and self.scheduler_manager.spills_at_reuse
        )

    def on_held_allocation(
        self, request: "Request", loop: str, num_behind: int, may_skip: bool
    ) -> bool:
        """The spill guard held `request`'s allocation: whether the
        scheduler's loop skips it instead of ending."""
        return self.scheduler_manager is not None and (
            self.scheduler_manager.on_held_allocation(
                request, loop, num_behind, may_skip
            )
        )

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished(request, block_ids)
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished_all_groups(
                request, block_ids
            )
        return False, None

    def has_pending_push_work(self) -> bool:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.has_pending_stores()
        return False

    def get_boundary_store_stats(self) -> BoundaryStoreStats | None:
        """Return cumulative boundary-handoff store diagnostics."""
        if self.scheduler_manager is not None:
            return self.scheduler_manager.get_boundary_store_stats()
        return None

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        # Worker-side callers may invoke this hook on connectors constructed
        # via __new__ (as in test_worker.py), which lack scheduler_manager.
        scheduler_manager: SimpleCPUOffloadScheduler | None = getattr(
            self, "scheduler_manager", None
        )
        if scheduler_manager is not None:
            return scheduler_manager.get_stats()
        return None

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return (
            SimpleCPUOffloadStats(data=data)
            if data is not None
            else SimpleCPUOffloadStats()
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return SimpleCPUOffloadPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

    def take_events(self) -> Iterable[KVCacheEvent]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.take_events()
        return []

    # NOTE: Workers are not contacted. In-flight transfers drain naturally,
    # and stale completions are ignored by the guarded
    # SimpleCPUOffloadScheduler._process_store_event().
    def reset_cache(self) -> bool | None:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.reset()
        return None
