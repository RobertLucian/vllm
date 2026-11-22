# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The spill guard holds even when RAM has no block to copy into.

The guard used to let held allocations go once RAM had no free block (all
pinned by restores or taken by stores in flight). With a RAM pool smaller
than the agents' sessions, that happens in bursts. RAM may then evict whole
sessions (their next turns re-read them, by design), but no allocation may
take a block the guard holds back (`unspilled_evicted_blocks`), and the
agents must keep making progress (no stall).
"""

import pytest

from .checks import delta
from .engine import EngineSpec
from .workloads import Session, drive

SPECS = [
    EngineSpec(
        family="gdn_fat", connector="tiered", gpu_blocks=64, ram_blocks=32, tp=tp
    )
    for tp in (1, 2)
] + [EngineSpec(family="dense", connector="tiered", gpu_blocks=64, ram_blocks=32)]

_AGENTS = 6
_SECONDS = 30.0


@pytest.mark.parametrize("engine", SPECS, indirect=True, ids=lambda s: s.id)
def test_full_ram_keeps_the_guard(engine, record_property):
    probe, llm, spec = engine.probe, engine.llm, engine.spec
    sessions = [
        Session.start(
            seed=300 + i,
            vocab_size=spec.model.vocab_size,
            block_size=probe.block_size,
            num_blocks=3,
            unit=probe.hash_block_size,
        )
        for i in range(_AGENTS)
    ]
    before = probe.counters()
    turns = drive(
        llm,
        sessions,
        seconds=_SECONDS,
        turn_blocks=1,
        resume_unit=probe.resume_unit,
        max_blocks=12,
        restart_blocks=3,
    )
    changed = delta(before, probe.counters())
    record_property("turns", len(turns))
    for name, value in sorted(changed.items()):
        record_property(name, value)

    assert len(turns) >= 2 * _AGENTS, f"agents stalled: {len(turns)} turns"
    assert changed.get("spill.unspilled_evicted_blocks", 0) == 0, changed
