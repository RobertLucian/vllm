# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One description of an engine under test, and its lifetime.

`EngineSpec` maps to `vllm.LLM` arguments; equal specs share one engine. The
engine core runs in this process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, set by
conftest.py), so tests read the scheduler and both block pools directly, and
the workers' tensors through `LLM.collective_rpc` at any tensor-parallel size.
"""

from __future__ import annotations

import dataclasses
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from vllm import LLM
from vllm.config import KVTransferConfig

from .models import FAMILIES, ModelFamily
from .probes import Probe

# Upper bounds on each tier's bytes; `gpu_blocks` and `ram_blocks` set their
# sizes.
_KV_BYTES_BOUND = 8 << 30
_CPU_BYTES_BOUND = 64 << 30
# RAM GDSF's prefill cost model: production's fit (Qwen3.8-27B, 2x RTX 3090).
# It only ranks sessions, so any model's tests can share it.
_PREFILL_COST = (7.4893e-04, 6.4382e-04)

Connector = Literal["none", "eager", "lazy", "tiered"]


@dataclass(frozen=True)
class EngineSpec:
    family: str = "gdn"
    # "eager": SimpleCPU's mirror (upstream main's design). "lazy": SimpleCPU
    # copying near eviction. "tiered": the fork's connector.
    connector: Connector = "tiered"
    ram_eviction: Literal["lru", "gdsf"] = "lru"
    gpu_blocks: int = 64
    ram_blocks: int = 128
    copy_ahead_blocks: int = 0
    # RAM blocks a restore loads at a time when the GPU lacks room for all of
    # it (0: the whole session at once).
    restore_chunk_blocks: int = 16
    # A session's checkpoint ladder: its first rung this many blocks back from
    # its end (0: none).
    ram_ladder_blocks: int = 8
    # RAM GDSF: a session idle this many times its turn gap (at least the
    # floor) is abandoned (0: off).
    abandon_gap_multiplier: float = 2.5
    abandon_floor_s: float = 30.0
    ram_state_spacing_blocks: int = 0
    prune_gpu_states: bool = True
    # Exact-end and floor states (hybrids only).
    frontier: bool = True
    prefix_match_unit: int | None = 16
    tp: int = 1
    # -1: the longest request `gpu_blocks` can hold.
    max_model_len: int = -1
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 8
    # > 0: priority scheduling with this many running slots kept for chats.
    chat_slots: int = 0
    low_priority_prefill_tokens: int | None = None
    async_scheduling: bool = False
    cuda_graphs: bool = False
    seed: int = 0

    @property
    def model(self) -> ModelFamily:
        return FAMILIES[self.family]

    def replace(self, **changes: Any) -> EngineSpec:
        return dataclasses.replace(self, **changes)

    @property
    def id(self) -> str:
        """A test id: family, connector, TP and what differs from the
        defaults."""
        default = EngineSpec()
        parts = [self.family, self.connector, f"tp{self.tp}"]
        for f in dataclasses.fields(self):
            if f.name in ("family", "connector", "tp"):
                continue
            value = getattr(self, f.name)
            if value != getattr(default, f.name):
                parts.append(f"{f.name}={value}")
        return "-".join(parts)

    def llm_kwargs(self, model_dir: Path) -> dict[str, Any]:
        model = self.model
        kwargs: dict[str, Any] = dict(
            model=str(model_dir),
            skip_tokenizer_init=True,
            load_format="dummy" if model.weights == "dummy" else "auto",
            dtype="bfloat16",
            seed=self.seed,
            enforce_eager=not self.cuda_graphs,
            tensor_parallel_size=self.tp,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            enable_prefix_caching=True,
            # Finer than a block only with several KV cache groups (vLLM
            # refuses it for one).
            prefix_match_unit=self.prefix_match_unit
            if len(model.layouts) > 1
            else None,
            num_gpu_blocks_override=self.gpu_blocks,
            # A fixed KV budget skips memory profiling, which fails when another
            # engine on the same GPU frees memory meanwhile. `gpu_blocks` sets
            # what is allocated; this only bounds it.
            kv_cache_memory_bytes=_KV_BYTES_BOUND,
            gpu_memory_utilization=0.2,
            async_scheduling=self.async_scheduling,
            disable_log_stats=True,
        )
        if model.hybrid:
            kwargs.update(
                mamba_cache_mode="align",
                # Restored states must equal recomputed ones bit for bit.
                mamba_ssm_cache_dtype="float32",
                mamba_frontier_checkpoint=self.frontier,
            )
        if model.mtp:
            kwargs["speculative_config"] = {
                "method": "mtp",
                "num_speculative_tokens": 2,
                "disable_eagle_block_drop": True,
                "truncate_acceptance_at_stop": True,
            }
        if self.chat_slots:
            kwargs.update(
                scheduling_policy="priority",
                reserved_priority_slots=self.chat_slots,
            )
            if self.low_priority_prefill_tokens is not None:
                kwargs["low_priority_prefill_tokens_per_step"] = (
                    self.low_priority_prefill_tokens
                )
        if self.connector != "none":
            kwargs["kv_transfer_config"] = self._kv_transfer_config()
        return kwargs

    def _kv_transfer_config(self) -> KVTransferConfig:
        extra: dict[str, Any] = {
            "cpu_bytes_to_use": _CPU_BYTES_BOUND,
            "num_cpu_blocks": self.ram_blocks,
        }
        if self.connector == "tiered":
            name = "TieredCPUOffloadConnector"
            extra.update(
                ram_eviction=self.ram_eviction,
                ram_state_spacing_blocks=self.ram_state_spacing_blocks,
                prune_gpu_states=self.prune_gpu_states,
                copy_ahead_blocks=self.copy_ahead_blocks,
                restore_chunk_blocks=self.restore_chunk_blocks,
                ram_ladder_blocks=self.ram_ladder_blocks,
                abandon_gap_multiplier=self.abandon_gap_multiplier,
                abandon_floor_seconds=self.abandon_floor_s,
            )
            if self.ram_eviction == "gdsf":
                extra["prefill_cost_coefficients"] = list(_PREFILL_COST)
        else:
            name = "SimpleCPUOffloadConnector"
            extra["lazy_offload"] = self.connector == "lazy"
        return KVTransferConfig(
            kv_connector=name,
            kv_role="kv_both",
            # Frontier checkpoints refuse any other policy.
            kv_load_failure_policy="recompute",
            kv_connector_extra_config=extra,
        )


class Engine:
    """A live `LLM` built from a spec, with its probe."""

    def __init__(self, spec: EngineSpec, model_dir: Path):
        self.spec = spec
        started = time.monotonic()
        self.llm = LLM(**spec.llm_kwargs(model_dir))
        self.boot_seconds = time.monotonic() - started
        self.probe = Probe(self.llm)

    def reset(self) -> None:
        """Empty both tiers between tests: in-flight copies land first."""
        self.probe.wait_for_copies()
        deadline = time.monotonic() + 60
        while not self.llm.reset_prefix_cache(reset_connector=True):
            assert time.monotonic() < deadline, "prefix caches did not reset"
            self.llm.llm_engine.step()

    def shutdown(self) -> None:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        self.llm.llm_engine.engine_core.shutdown()
        del self.llm
        gc.collect()
        cleanup_dist_env_and_memory()
