# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every model family boots with every CPU tier and resumes a turn.

A family or tensor-parallel size a connector cannot serve fails here by
name. Each case records its geometry (block size and bytes, boot time, free
GPU memory) as JUnit properties.
"""

import pytest
import torch

from .engine import EngineSpec
from .models import FAMILIES
from .workloads import Session, run

SPECS = [
    EngineSpec(family=family, connector=connector)
    for family in FAMILIES
    for connector in ("none", "eager", "tiered")
] + [
    EngineSpec(family=family, connector="tiered", tp=tp)
    for family in ("gdn", "dense", "swa")
    for tp in (2, 4, 8)
]


@pytest.mark.parametrize("engine", SPECS, indirect=True, ids=lambda s: s.id)
def test_boots_and_resumes(engine, record_property):
    probe, llm, spec = engine.probe, engine.llm, engine.spec
    record_property("boot_seconds", round(engine.boot_seconds, 1))
    record_property("block_size", probe.block_size)
    record_property("prefix_match_unit", probe.hash_block_size)
    record_property("num_gpu_blocks", probe.num_gpu_blocks)
    record_property("kv_groups", len(probe.group_block_sizes))
    if spec.connector != "none":
        record_property("block_bytes_per_rank", probe.block_bytes())
    free, total = torch.cuda.mem_get_info()
    record_property("gpu_free_gib", round(free / 2**30, 2))

    session = Session.start(
        seed=1,
        vocab_size=spec.model.vocab_size,
        block_size=probe.block_size,
        num_blocks=3,
        unit=probe.hash_block_size,
    )
    first = run(llm, [session.prompt])[0]
    session.record(first)
    assert first.num_cached_tokens == 0

    second = run(llm, [session.next_tool_turn(0.5)])[0]
    record_property("resumed_at", second.num_cached_tokens)
    # The whole first prompt is cached, less at most one resume unit.
    end = len(first.prompt)
    assert end - probe.resume_unit <= second.num_cached_tokens <= end, probe.residency(
        second.prompt
    )
