# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU <-> pinned RAM copy bandwidth, per rank, through the connector's own
copy path: `copy_blocks` with the worker's store and load parameters on its
low-priority streams, over its real KV tensors.

It records GB/s (decimal) per direction as JUnit properties: the lazy tier
must copy a block in less time than the GPU takes to need it back.
"""

import subprocess

import pytest

from .engine import EngineSpec

# A spec of its own: the copies overwrite blocks 1..n of a fresh engine.
SPECS = [
    EngineSpec(family="gdn_fat", connector="tiered", ram_blocks=64, seed=1, tp=tp)
    for tp in (1, 2)
]


def _measure(worker, num_blocks: int, repeats: int) -> dict:
    import torch

    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm.v1.simple_kv_offload.cuda_mem_ops import copy_blocks

    handler = get_kv_transfer_group().worker_handler
    backend = handler._backend
    gpu_blocks = next(iter(handler.gpu_kv_caches.values())).shape[0]
    n = min(num_blocks, handler.num_cpu_blocks - 1, gpu_blocks - 1)
    # Block 0 is the null block: leave it alone.
    ids = list(range(1, n + 1))
    block_bytes = sum(t[0].nbytes for t in handler.gpu_kv_caches.values())
    result = {
        "device": torch.cuda.get_device_name(),
        "block_mib": round(block_bytes / 2**20, 2),
        "blocks": n,
    }
    for name, params, stream in (
        ("store_gbps", backend._store_params, handler.store_stream),
        ("load_gbps", backend._load_params, handler.load_stream),
    ):
        copy_blocks(ids, ids, params)
        stream.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(repeats):
            copy_blocks(ids, ids, params)
        end.record(stream)
        end.synchronize()
        seconds = start.elapsed_time(end) / 1e3
        result[name] = round(n * repeats * block_bytes / seconds / 1e9, 2)
    return result


@pytest.mark.parametrize("engine", SPECS, indirect=True, ids=lambda s: s.id)
def test_dma_bandwidth(engine, record_property):
    per_rank = engine.llm.collective_rpc(_measure, args=(48, 5))
    for rank, result in enumerate(per_rank):
        for key, value in result.items():
            record_property(f"rank{rank}_{key}", value)
        print(f"DMA rank {rank}: {result}")
        assert result["store_gbps"] > 0 and result["load_gbps"] > 0
    links = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.width.current",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    record_property("pcie_links", links)
    print(f"PCIe links: {links}")
    # The overwritten blocks hold no cached keys after this reset.
    engine.reset()
