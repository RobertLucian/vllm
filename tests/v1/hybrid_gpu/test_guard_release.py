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

Nor does the guard let go while a copy is in flight. Its wall-clock limit
(`_MAX_SPILL_WAIT_S`) used to run from the hold's start whatever was in
flight, and a copy's acknowledgement takes scheduling passes, each as long
as its step's prefill (GPU window B, 2026-11-25: 11 blocks left uncopied
with 32 in flight). The limit now applies only while no copy is in flight.
The last test sets it to 0 on pools with RAM to spare: every hold would end
at once under the old rule, and here none may end with a copy in flight
(`guard_releases_in_flight`). A hold with none in flight and nothing left to
copy still ends at once at that limit: `guard_releases` and what such
releases cost are recorded, not asserted.
"""

import time

import pytest

from vllm.v1.simple_kv_offload import manager as offload_manager

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


def test_guard_holds_while_a_copy_is_in_flight(engines, monkeypatch, record_property):
    # The scheduler runs in this process and reads the limit at every walk.
    monkeypatch.setattr(offload_manager, "_MAX_SPILL_WAIT_S", 0.0)
    # Production's RAM policy and scheduling, with RAM for every session
    # (test_spill_safety's pools): nothing but a release loses a block.
    spec = EngineSpec(
        family="gdn_fat",
        connector="tiered",
        ram_eviction="gdsf",
        async_scheduling=True,
        gpu_blocks=96,
        ram_blocks=384,
    )
    engine = engines.get(spec)
    probe, llm = engine.probe, engine.llm
    sessions = [
        Session.start(
            seed=350 + i,
            vocab_size=spec.model.vocab_size,
            block_size=probe.block_size,
            num_blocks=4,
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
        user_every=5,
        max_blocks=24,
    )
    changed = delta(before, probe.counters())
    short = [t for t in turns if t.short_by > 0]
    record_property("turns", len(turns))
    record_property("short_resumes", len(short))
    for name, value in sorted(changed.items()):
        record_property(name, value)

    # The run did what it checks: allocations waited for copies. With the
    # limit at 0 a hold with nothing in flight and nothing to store lets go
    # at once (a block may leave uncopied then: recorded above); one with a
    # copy in flight never does.
    assert changed.get("spill.held_allocations", 0) > 0, changed
    assert changed.get("spill.guard_releases_in_flight", 0) == 0, changed
