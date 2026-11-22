# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only views of a live engine: nothing here replaces or patches it.

Scheduler side: the scheduler, its GPU block pool and, with a CPU connector,
the connector's scheduler half and RAM pool, read in this process. Worker
side: the KV tensors in VRAM and pinned RAM, read on every rank through
`LLM.collective_rpc` (a rank holds its shard of each block).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm import SamplingParams
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm import LLM


def _worker_digests(worker: Any, tier: str, block_ids: list[int]) -> list[str]:
    """On a worker: a digest of each block's bytes across all its KV tensors."""
    import hashlib

    from vllm.distributed.kv_transfer import get_kv_transfer_group

    handler = get_kv_transfer_group().worker_handler
    caches = handler.gpu_kv_caches if tier == "gpu" else handler.cpu_kv_caches
    digests = []
    for block_id in block_ids:
        h = hashlib.blake2b(digest_size=16)
        for name in sorted(caches):
            h.update(caches[name][block_id].cpu().numpy().tobytes())
        digests.append(h.hexdigest())
    return digests


@dataclass(frozen=True)
class Residency:
    """Where each of a token sequence's keys is cached, per KV cache group,
    one mark per prefix-match unit: B both tiers, G GPU only, C RAM only,
    . neither (the format of the connector's `describe_residency`)."""

    marks: dict[int, str]
    units_per_block: dict[int, int]

    def blocks(self, group_id: int) -> str:
        """The marks at the group's block ends: one per block."""
        per = self.units_per_block[group_id]
        return self.marks[group_id][per - 1 :: per]

    def __str__(self) -> str:
        return " | ".join(f"g{g} {m}" for g, m in self.marks.items())


class Probe:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.core = llm.llm_engine.engine_core.engine_core
        self.scheduler = self.core.scheduler
        self.gpu_pool = self.scheduler.kv_cache_manager.block_pool
        connector = self.scheduler.connector
        self.tier = connector.scheduler_manager if connector is not None else None
        self.cpu_pool = self.tier.cpu_block_pool if self.tier is not None else None
        groups = self.scheduler.kv_cache_config.kv_cache_groups
        self.group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
        self.full_attention_group_ids = [
            i
            for i, g in enumerate(groups)
            if isinstance(g.kv_cache_spec, FullAttentionSpec)
        ]
        self.mamba_group_ids = [
            i for i, g in enumerate(groups) if isinstance(g.kv_cache_spec, MambaSpec)
        ]
        self.sliding_window_group_ids = [
            i
            for i, g in enumerate(groups)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec)
        ]
        self.hash_block_size = (
            self.tier.hash_block_size
            if self.tier is not None
            else self.scheduler.cache_config.prefix_match_unit
            or min(self.group_block_sizes)
        )

    # ------------------------------------------------------------------
    # Geometry

    @property
    def block_size(self) -> int:
        """Tokens per block of the full-attention group."""
        return max(self.group_block_sizes)

    @property
    def num_gpu_blocks(self) -> int:
        return self.gpu_pool.num_gpu_blocks

    @property
    def resume_unit(self) -> int:
        """How far short of its last prompt end a turn may resume: one
        prefix-match unit (an exact-end state sits a unit short of the end),
        or a block where hits land on block boundaries: a hybrid without
        exact-end states, and a sliding window, whose groups key whole blocks
        only (window A: 128 of a 192-token prompt, 64-token blocks)."""
        if self.sliding_window_group_ids or (
            self.mamba_group_ids
            and not self.scheduler.cache_config.mamba_frontier_checkpoint
        ):
            return self.block_size
        return self.hash_block_size

    def block_bytes(self) -> list[int]:
        """Bytes one block takes on each rank."""

        def count(worker: Any) -> int:
            from vllm.distributed.kv_transfer import get_kv_transfer_group

            caches = get_kv_transfer_group().worker_handler.gpu_kv_caches
            return sum(t[0].nbytes for t in caches.values())

        return self.llm.collective_rpc(count)

    # ------------------------------------------------------------------
    # Keys and residency

    def block_hashes(self, tokens: list[int]) -> list:
        """The prefix-cache keys of `tokens`, one per prefix-match unit."""
        request = Request(
            "probe",
            prompt_token_ids=list(tokens),
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=self.core.request_block_hasher,
        )
        return list(request.block_hashes)

    def residency(self, tokens: list[int]) -> Residency:
        hashes = self.block_hashes(tokens)
        gpu = self.gpu_pool.cached_block_hash_to_block
        cpu = self.cpu_pool.cached_block_hash_to_block if self.cpu_pool else None
        marks = {}
        for group_id in range(len(self.group_block_sizes)):
            line = []
            for block_hash in hashes:
                key = make_block_hash_with_group_id(block_hash, group_id)
                on_gpu = gpu.get_one_block(key) is not None
                in_ram = cpu is not None and cpu.get_one_block(key) is not None
                line.append(
                    "B"
                    if on_gpu and in_ram
                    else "G"
                    if on_gpu
                    else "C"
                    if in_ram
                    else "."
                )
            marks[group_id] = "".join(line)
        units = {
            g: size // self.hash_block_size
            for g, size in enumerate(self.group_block_sizes)
        }
        return Residency(marks, units)

    def block_ids(
        self, tokens: list[int], group_id: int, tier: str
    ) -> list[int | None]:
        """Per block of `group_id`, the id of the block caching it in `tier`
        ("gpu" or "ram"), or None."""
        hashes = self.block_hashes(tokens)
        per = self.group_block_sizes[group_id] // self.hash_block_size
        pool = self.gpu_pool if tier == "gpu" else self.cpu_pool
        assert pool is not None
        cached = pool.cached_block_hash_to_block
        out = []
        for block_hash in hashes[per - 1 :: per]:
            block = cached.get_one_block(
                make_block_hash_with_group_id(block_hash, group_id)
            )
            out.append(None if block is None else block.block_id)
        return out

    def cached_units(
        self, tokens: list[int], group_id: int, tier: str
    ) -> dict[int, int]:
        """Prefix-match unit -> block id, for every unit of `tokens` whose key
        `group_id` caches in `tier`: for a Mamba group, where its states sit
        (a finished turn's exact end is one unit short of its prompt end)."""
        pool = self.gpu_pool if tier == "gpu" else self.cpu_pool
        assert pool is not None
        cached = pool.cached_block_hash_to_block
        out = {}
        for unit, block_hash in enumerate(self.block_hashes(tokens)):
            block = cached.get_one_block(
                make_block_hash_with_group_id(block_hash, group_id)
            )
            if block is not None:
                out[unit] = block.block_id
        return out

    # ------------------------------------------------------------------
    # Bytes

    def digests(self, tier: str, block_ids: list[int]) -> list[tuple[str, ...]]:
        """Per block, the digest of its bytes on each rank."""
        per_rank = self.llm.collective_rpc(_worker_digests, args=(tier, block_ids))
        return list(zip(*per_rank))

    # ------------------------------------------------------------------
    # Counters and waiting

    def counters(self) -> dict[str, int]:
        """The connector's cumulative counters, flattened."""
        tier = self.tier
        if tier is None:
            return {}
        out: dict[str, int] = {}
        for prefix, stats in (
            ("spill", getattr(tier, "lazy_spill_stats", None)),
            ("restore", getattr(tier, "restore_stats", None)),
            ("ram", getattr(getattr(tier, "ram_policy", None), "stats", None)),
        ):
            if stats is not None:
                out.update(
                    {
                        f"{prefix}.{k}": v
                        for k, v in vars(stats).items()
                        if isinstance(v, int)
                    }
                )
        for outcome, count in getattr(tier, "prune_outcomes", {}).items():
            out[f"prune.{outcome}"] = count
        return out

    def wait_for_copies(self, timeout_s: float = 60.0) -> None:
        """Step the engine until no store is queued or in flight."""
        tier = self.tier
        if tier is None:
            return
        deadline = time.monotonic() + timeout_s
        while tier.has_pending_stores():
            assert time.monotonic() < deadline, "stores did not finish"
            self.llm.llm_engine.step()
