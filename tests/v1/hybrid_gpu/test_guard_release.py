# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The spill guard holds even when RAM has no block to copy into.

The guard used to let held allocations go once RAM had no free block (all
pinned by restores or taken by stores in flight), and after 10 s of some
allocation being held, which steady load always reaches. With a RAM pool
smaller than the agents' sessions, both happen in bursts. RAM may then evict
whole sessions (their next turns re-read them, by design), but no allocation
may take a block the guard holds back (`unspilled_evicted_blocks`), and the
agents must keep making progress: no stall, and no livelock. Holding until
copies landed (window D) livelocked at TP 1: RAM evicted the copies the held
allocations waited for to take new ones, so copies kept landing and the 30 s
run took 294 s.
"""

import time

import pytest

from .checks import delta
from .engine import EngineSpec
from .workloads import Session, drive

SPECS = (
    [
        EngineSpec(
            family="gdn_fat", connector="tiered", gpu_blocks=64, ram_blocks=32, tp=tp
        )
        for tp in (1, 2)
    ]
    + [
        # Production's RAM policy and scheduling.
        EngineSpec(
            family="gdn_fat",
            connector="tiered",
            ram_eviction="gdsf",
            async_scheduling=True,
            gpu_blocks=64,
            ram_blocks=32,
            tp=tp,
        )
        for tp in (1, 2)
    ]
    + [EngineSpec(family="dense", connector="tiered", gpu_blocks=64, ram_blocks=32)]
)

_AGENTS = 6
_SECONDS = 30.0
# The last turns finish after `_SECONDS`; a livelock takes several times it.
_MAX_SECONDS = 3 * _SECONDS


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
    started = time.monotonic()
    turns = drive(
        llm,
        sessions,
        seconds=_SECONDS,
        turn_blocks=1,
        resume_unit=probe.resume_unit,
        max_blocks=12,
        restart_blocks=3,
    )
    elapsed = time.monotonic() - started
    changed = delta(before, probe.counters())
    record_property("turns", len(turns))
    record_property("seconds", round(elapsed, 1))
    for name, value in sorted(changed.items()):
        record_property(name, value)

    assert len(turns) >= 2 * _AGENTS, f"agents stalled: {len(turns)} turns"
    assert elapsed <= _MAX_SECONDS, f"livelock: {elapsed:.0f} s, {changed}"
    assert changed.get("spill.unspilled_evicted_blocks", 0) == 0, changed
