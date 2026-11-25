# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A hybrid scheduler (an attention group, align-mode Mamba groups and an
indexer ring) and a step driver for CPU tests of hybrid state checkpoints."""

from typing import Any, Literal

import pytest
import torch

from tests.v1.kv_connector.unit.utils import MockKVConnector
from vllm.config import (
    CacheConfig,
    KVEventsConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.kv_transfer import KVRole
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

EOS = 2


class HybridMockKVConnector(MockKVConnector, SupportsHMA):
    """`MockKVConnector` (a fixed number of matched tokens, loads nothing)
    on several KV cache groups."""

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None


KVConnectorFactory.register_connector(
    "HybridMockKVConnector", __name__, HybridMockKVConnector.__name__
)


def mock_connector(
    matched_tokens: int,
    kv_role: KVRole = "kv_both",
    kv_load_failure_policy: Literal["recompute", "fail"] = "recompute",
) -> KVTransferConfig:
    """A connector that matches `matched_tokens` more tokens of every request
    and loads them synchronously."""
    return KVTransferConfig(
        kv_connector="HybridMockKVConnector",
        kv_role=kv_role,
        kv_load_failure_policy=kv_load_failure_policy,
        kv_connector_extra_config={"matched_tokens": matched_tokens, "is_async": False},
    )


def simple_cpu_connector(num_cpu_blocks: int, lazy: bool = False) -> KVTransferConfig:
    """The SimpleCPU offloading connector with a CPU pool of `num_cpu_blocks`,
    storing every computed block (eager) or only blocks about to be evicted
    (lazy)."""
    return KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_load_failure_policy="recompute",
        kv_connector_extra_config={
            "num_cpu_blocks": num_cpu_blocks,
            "cpu_bytes_to_use": 1 << 40,
            "lazy_offload": lazy,
        },
    )


def tiered_cpu_connector(
    num_cpu_blocks: int,
    ram_eviction: str = "lru",
    ram_state_spacing_blocks: int = 8,
    copy_ahead_blocks: int = 0,
) -> KVTransferConfig:
    """The tiered connector: a RAM pool of `num_cpu_blocks` that extends VRAM,
    restoring into the GPU prefix cache before each lookup."""
    return KVTransferConfig(
        kv_connector="TieredCPUOffloadConnector",
        kv_role="kv_both",
        kv_load_failure_policy="recompute",
        kv_connector_extra_config={
            "num_cpu_blocks": num_cpu_blocks,
            "cpu_bytes_to_use": 1 << 40,
            "ram_eviction": ram_eviction,
            "prefill_cost_coefficients": [1.862e-5, 3.33e-6],
            "ram_state_spacing_blocks": ram_state_spacing_blocks,
            "copy_ahead_blocks": copy_ahead_blocks,
            # This harness models neither DMA time nor production's Mamba
            # retention: chunked restores are tested on CUDA
            # (tests/v1/hybrid_gpu).
            "restore_chunk_blocks": 0,
            # Its dense Mamba retention keeps every chunk's state, so rungs
            # would only shift its sessions' points; tested on CUDA too.
            "ram_ladder_blocks": 0,
            # Its fake clock has no turn gaps to speak of.
            "abandon_gap_multiplier": 0,
            # Its tests were written for the restore wait and the queue behind
            # it; pack admission is tested on CUDA.
            "admission": "legacy",
        },
    )


def make_hybrid_scheduler(
    block_size: int = 64,
    num_blocks: int = 200,
    max_num_batched_tokens: int = 1024,
    long_prefill_token_threshold: int = 0,
    checkpoint_alignment: int | None = 16,
    ring: int = 4,
    resume_catchup: str | None = "decode",
    hash_block_size: int | None = None,
    kv_transfer_config: KVTransferConfig | None = None,
    speculative_config: SpeculativeConfig | None = None,
    num_speculative_blocks: int = 0,
    ring_compression_ratio: int | None = None,
    async_scheduling: bool = False,
    kv_events_config: KVEventsConfig | None = None,
    **cache_kwargs: Any,
) -> Scheduler:
    """`cache_kwargs` are CacheConfig fields; `ring` 0 builds no ring.
    `num_speculative_blocks`: the Mamba groups' state columns per draft
    position."""
    vllm_config = VllmConfig(
        kv_transfer_config=kv_transfer_config,
        kv_events_config=kv_events_config,
        speculative_config=speculative_config,
        model_config=ModelConfig(
            model="facebook/opt-125m",
            dtype="float16",
            seed=42,
            skip_tokenizer_init=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=8192,
            long_prefill_token_threshold=long_prefill_token_threshold,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=async_scheduling,
        ),
        cache_config=CacheConfig(
            block_size=block_size,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            **cache_kwargs,
        ),
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    groups = [
        KVCacheGroupSpec(
            ["attn"],
            FullAttentionSpec(
                block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        *(
            KVCacheGroupSpec(
                [f"mamba{i}"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_prefill_checkpoint_blocks=int(checkpoint_alignment is not None),
                    prefill_checkpoint_alignment=checkpoint_alignment,
                    num_speculative_blocks=num_speculative_blocks,
                ),
            )
            for i in range(2)
        ),
    ]
    if ring:
        groups.append(
            KVCacheGroupSpec(
                ["ring"],
                CircularBufferSpec(
                    block_size=ring,
                    num_kv_heads=2,
                    head_size=1,
                    head_size_v=0,
                    dtype=torch.bfloat16,
                    dcp_sharded=False,
                    resume_catchup=resume_catchup,
                    compression_ratio=ring_compression_ratio,
                ),
            )
        )
    tensors = []
    if kv_transfer_config is not None:
        # One backing allocation that every group overlays, as on the GPU: a
        # connector sizes its own pool from it.
        page = max(group.kv_cache_spec.page_size_bytes for group in groups)
        tensors = [
            KVCacheTensor(
                size=page * num_blocks,
                layers=list(group.layer_names),
                layer_stride=page * num_blocks,
                block_stride=page,
            )
            for group in groups
        ]
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=groups
    )
    register_all_kvcache_specs(vllm_config)
    # The checkpoint features need Model Runner V2, which CPU builds select
    # only when asked to.
    scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
    with pytest.MonkeyPatch.context() as m:
        m.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        return scheduler_cls(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=block_size,
            hash_block_size=hash_block_size or block_size,
            log_stats=True,
        )


def prompt(seed: int, num_tokens: int) -> list[int]:
    return [3 + (seed * 7919 + i * 104729) % 30000 for i in range(num_tokens)]


def make_request(
    scheduler: Scheduler,
    request_id: str,
    tokens: list[int],
    max_tokens: int,
) -> Request:
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, EOS)
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(scheduler.hash_block_size, sha256),
    )


def add_request(
    scheduler: Scheduler,
    request_id: str,
    tokens: list[int],
    max_tokens: int,
) -> Request:
    request = make_request(scheduler, request_id, tokens, max_tokens)
    scheduler.add_request(request)
    return request


def step(scheduler: Scheduler) -> SchedulerOutput:
    """Schedule one step and return its output; requests that finished their
    prefill sample one token."""
    output = scheduler.schedule()
    req_ids = list(output.num_scheduled_tokens)
    sampled = []
    for request_id in req_ids:
        request = scheduler.requests[request_id]
        position = request.num_computed_tokens
        sampled.append([] if request.is_prefill_chunk else [3 + position % 997])
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return output


def run(scheduler: Scheduler, *requests: Request) -> list[SchedulerOutput]:
    outputs = []
    while not all(r.is_finished() for r in requests):
        outputs.append(step(scheduler))
    return outputs


def admitted_at(scheduler: Scheduler, request: Request) -> int:
    """Step until `request` is admitted; returns the tokens its prefix-cache
    hit covered."""
    while request.status == RequestStatus.WAITING:
        output = step(scheduler)
        for new in output.scheduled_new_reqs:
            if new.req_id == request.request_id:
                return new.num_computed_tokens
    raise AssertionError(f"{request.request_id} was not admitted")
