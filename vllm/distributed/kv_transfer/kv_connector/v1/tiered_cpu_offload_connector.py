# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TieredCPUOffloadConnector: a RAM tier that extends VRAM.

One connector owns everything that moves KV between VRAM and RAM:
- What leaves VRAM: the GPU prefix cache's LRU order (``--prefix-cache-eviction
  lru``, required), except that what queued requests resume from goes last,
  in either tier.
- How it leaves: copied to RAM before the GPU may drop it, with the other
  groups' blocks under the same keys (SimpleCPU's lazy walk and spill guard),
  ``copy_ahead_blocks`` at the eviction end ahead of need. Recurrent states at
  block boundaries are copied only where live sessions resume, where shared
  prefixes end (a system prompt sessions start with), and every
  ``ram_state_spacing_blocks`` blocks if set; with ``prune_gpu_states``
  (default) the GPU releases the others when a request finishes.
- What leaves RAM: ``ram_eviction`` (``lru`` or ``gdsf``).
- How it comes back: restored into the GPU prefix cache before the request's
  lookup, which then hits it like any GPU-resident prefix
  (vllm/v1/simple_kv_offload/tiered.py).

Configuration (``kv_connector_extra_config``): SimpleCPU's ``cpu_bytes_to_use``
/ ``num_cpu_blocks``, plus ``ram_eviction``, ``prefill_cost_coefficients``
(``[D, E]``, seconds per token and per token per 100k of context, used by
``gdsf``), ``frequency_half_life_seconds``, ``ram_state_spacing_blocks``
(default 0, off), ``prune_gpu_states`` (default true),
``copy_ahead_blocks`` (default 0: the lazy walk's own target),
``restore_chunk_blocks`` (default 16: a restore the GPU lacks room for loads
in chunks of this many blocks, a swap; 0 restores whole),
``spill`` (default ``walk``: the lazy walk copies a block ahead of its
eviction, and the GPU reuses it once the copy is acknowledged, which holds
allocations for scheduling passes; ``inline``: a block RAM lacks is copied
in the step that reuses it, ordered on the device before anything in that
step writes it, so nothing is held or acknowledged; RAM in host memory
only),
``ram_ladder_blocks`` (default 8: a session keeps a state per doubling
stretch from this many blocks back from its end, so RAM evicts its tail down
to a rung, not all of it; 0, off), ``ram_ladder_ranking`` (default ``fixed``:
rungs go only with the tail above them; with ``gdsf``, ``safe``: a rung no
cut could stop at goes at once and the others are ranked with the pages per
state block; ``expected``: by what losing one adds to the session's expected
recompute) and ``abandon_gap_multiplier`` /
``abandon_floor_seconds`` (default 2.5 / 30: RAM GDSF evicts a session idle
that many times its turn gap, at least the floor, before any live one; 0,
off), ``admission`` (default ``legacy``; ``pack``, ``pass``: window T, 2026-11-24,
had pack below legacy at TP 2) and
``claim_after_seconds`` (default 1), ``bounded_search`` (default true: a
lookup searches no deeper than the restore gain it checks; false walks the
whole session), ``verify_search`` (default false: each bounded lookup is
checked against the whole walk, for tests), ``log_restores`` (default false:
each restore's start, landing and admission logged at INFO, else DEBUG) and
``summary_log_seconds`` (default 60: a summary line at most this often after
activity; 0, off).
"""

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    PromMetric,
    PromMetricT,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)

from vllm.v1.simple_kv_offload.metrics import TieredCPUOffloadPromMetrics
from vllm.v1.simple_kv_offload.tiered import TieredCPUOffloadScheduler

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class TieredCPUOffloadConnector(SimpleCPUOffloadConnector):
    """A RAM tier that extends VRAM (module docstring)."""

    scheduler_manager_cls = TieredCPUOffloadScheduler
    always_lazy = True

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        # Restores land in the GPU prefix cache before the lookup: every hit
        # is a local one.
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        policy = vllm_config.cache_config.prefix_cache_eviction
        if policy != "lru":
            raise ValueError(
                "TieredCPUOffloadConnector moves blocks from VRAM to RAM in the "
                "GPU prefix cache's LRU order; use --prefix-cache-eviction lru "
                f"(got {policy!r}). Its RAM policy is ram_eviction."
            )
        super().__init__(vllm_config, role, kv_cache_config)

    def _scheduler_manager_kwargs(self, extra_config: dict[str, Any]) -> dict[str, Any]:
        coefficients = extra_config.get("prefill_cost_coefficients")
        return {
            "ram_eviction": str(extra_config.get("ram_eviction", "lru")),
            "prefill_cost_coefficients": (
                tuple(float(c) for c in coefficients) if coefficients else None
            ),
            "frequency_half_life_s": float(
                extra_config.get("frequency_half_life_seconds", 600.0)
            ),
            "ram_state_spacing_blocks": int(
                extra_config.get("ram_state_spacing_blocks", 0)
            ),
            "prune_gpu_states": bool(extra_config.get("prune_gpu_states", True)),
            "copy_ahead_blocks": int(extra_config.get("copy_ahead_blocks", 0)),
            "restore_chunk_blocks": int(extra_config.get("restore_chunk_blocks", 16)),
            "ram_ladder_blocks": int(extra_config.get("ram_ladder_blocks", 8)),
            "ram_ladder_ranking": str(extra_config.get("ram_ladder_ranking", "fixed")),
            "abandon_gap_multiplier": float(
                extra_config.get("abandon_gap_multiplier", 2.5)
            ),
            "abandon_floor_s": float(extra_config.get("abandon_floor_seconds", 30.0)),
            "admission": str(extra_config.get("admission", "legacy")),
            "claim_after_s": float(extra_config.get("claim_after_seconds", 1.0)),
            "bounded_search": bool(extra_config.get("bounded_search", True)),
            "verify_search": bool(extra_config.get("verify_search", False)),
            "log_restores": bool(extra_config.get("log_restores", False)),
            "summary_log_s": float(extra_config.get("summary_log_seconds", 60.0)),
        }

    def bind_kv_cache_manager(self, kv_cache_manager: "KVCacheManager") -> None:
        super().bind_kv_cache_manager(kv_cache_manager)
        manager = self.scheduler_manager
        if isinstance(manager, TieredCPUOffloadScheduler):
            manager.bind_kv_cache_manager(kv_cache_manager)

    def bind_first_chunk(
        self,
        first_chunk: Callable[["Request", int, int, tuple[int, ...]], tuple[int, int]],
    ) -> None:
        """The scheduler's sizing of a restored request's first chunk
        (`Scheduler.first_chunk`)."""
        manager = self.scheduler_manager
        if isinstance(manager, TieredCPUOffloadScheduler):
            manager.bind_first_chunk(first_chunk)

    def observe_admission(self, request: "Request", num_computed_tokens: int) -> None:
        """The scheduler admitted `request` `num_computed_tokens` tokens in."""
        manager = self.scheduler_manager
        if isinstance(manager, TieredCPUOffloadScheduler):
            manager.observe_admission(request, num_computed_tokens)

    def observe_queue(self, waiting: Iterable["Request"]) -> None:
        """The requests waiting for admission as a scheduler step starts."""
        manager = self.scheduler_manager
        if isinstance(manager, TieredCPUOffloadScheduler):
            manager.observe_queue(waiting)

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return TieredCPUOffloadPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )
