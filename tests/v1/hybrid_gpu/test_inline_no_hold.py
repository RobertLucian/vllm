# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""With inline copies nothing waits for a copy, and nothing is lost.

The lazy walk (`spill="walk"`) lets the GPU reuse a cached block only once
its copy to RAM is acknowledged: allocations are held meanwhile, restores
wait for the guard, and a lookup that finds its blocks' copies in flight
waits for them. With `spill="inline"` the block is copied in the step that
reuses it and RAM lists the copy at once. test_spill_safety's agents (eight,
out of phase, on a GPU pool far smaller than their sessions) run under
production's RAM policy and scheduling: no allocation may be held, no restore
or lookup may wait for a copy, blocks must have been copied inline, no block
a lookup could restore may leave uncopied, and every turn must resume where
its session's last prompt ended.

The same agents then run with the walk: both runs' turns are recorded side
by side, measured, not judged.
"""

import pytest

from .checks import delta, save_result
from .engine import EngineSpec
from .test_spill_safety import _AGENTS, _SECONDS
from .workloads import Session, Turn, drive

# What waiting for a copy counts, and what leaving without one does.
_ZERO = (
    "spill.held_allocations",
    "spill.hold_episodes",
    "spill.guard_releases",
    "spill.unspilled_evicted_blocks",
    "restore.waited_guard",
    "lookup.copies_restore",
)


def _spec(family: str, tp: int, spill: str) -> EngineSpec:
    return EngineSpec(
        family=family,
        connector="tiered",
        ram_eviction="gdsf",
        async_scheduling=True,
        gpu_blocks=96,
        ram_blocks=384,
        spill=spill,  # type: ignore[arg-type]
        tp=tp,
    )


def _agents(
    engines, spec: EngineSpec, record_property
) -> tuple[list[Turn], dict[str, int]]:
    """test_spill_safety's workload; the turns and what the counters did."""
    engine = engines.get(spec)
    probe, llm = engine.probe, engine.llm
    sessions = [
        Session.start(
            seed=100 + i,
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
    arm = spec.spill
    record_property(f"{arm}.turns", len(turns))
    record_property(f"{arm}.short_resumes", len(short))
    record_property(f"{arm}.recomputed_tokens", sum(t.short_by for t in short))
    for name, value in sorted(changed.items()):
        record_property(f"{arm}.{name}", value)
    save_result(
        "inline no_hold",
        spec.id,
        {"turns": len(turns), "short_resumes": len(short), "counters": changed},
    )
    return turns, changed


@pytest.mark.parametrize("tp", [1, 2], ids=lambda tp: f"tp{tp}")
@pytest.mark.parametrize("family", ["gdn_fat", "gdn_mtp"])
def test_nothing_waits_and_nothing_is_lost(engines, family, tp, record_property):
    turns, changed = _agents(engines, _spec(family, tp, "inline"), record_property)
    walk_turns, _ = _agents(engines, _spec(family, tp, "walk"), record_property)
    print(f"inline no-hold {family} tp{tp}: {len(turns)} turns, walk {len(walk_turns)}")

    waited = {name: changed[name] for name in _ZERO if changed.get(name, 0)}
    assert not waited, changed
    assert changed.get("spill.inline_spilled_blocks", 0) > 0, changed
    short = [t for t in turns if t.short_by > 0]
    assert not short, (
        f"{len(short)} of {len(turns)} turns resumed short of their session's "
        "last prompt end (session, floor, resumed): "
        f"{[(t.session, t.floor, t.result.num_cached_tokens) for t in short[:8]]}"
    )
