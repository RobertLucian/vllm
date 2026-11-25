# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack admission: restores side by side, admitted the step they land.

Under pack admission (`admission="pack"`; the default is legacy since
window T, 2026-11-24):
- a restore takes room for its request's tokens past the resume point
  besides the blocks it loads;
- every waiting restore the GPU has room for starts as soon as its copies
  land;
- the earliest request that cannot fit claims the room it waits for.

Three checks:
- sessions pushed to RAM and resumed together are each admitted in the first
  scheduling pass after their load lands, with the outputs of a full
  prefill;
- several restores start in one step when the GPU has room for them;
- a long session that restores in chunks is not starved by a stream of small
  turns.
"""

import pytest

from .checks import assert_outputs_close, delta, save_result
from .engine import EngineSpec
from .probes import Probe
from .workloads import Session, drive, fillers, greedy, run

# Most filler prompts it may take to push every session out of the GPU.
_MAX_FILLERS = 32


def _sessions(probe: Probe, spec: EngineSpec, count: int, num_blocks: float, seed: int):
    return [
        Session.start(
            seed=seed + i,
            vocab_size=spec.model.vocab_size,
            block_size=probe.block_size,
            num_blocks=num_blocks,
            unit=probe.hash_block_size,
        )
        for i in range(count)
    ]


def _push_to_ram(llm, probe: Probe, spec: EngineSpec, sessions: list[Session]) -> int:
    """Run filler prompts until every session has a page the GPU dropped:
    its next turn needs RAM. The fillers used."""
    fa_group = probe.full_attention_group_ids[0]
    used = 0
    for filler in fillers(11, _MAX_FILLERS, 8, spec.model.vocab_size, probe.block_size):
        if all(None in probe.block_ids(s.sequence, fa_group, "gpu") for s in sessions):
            return used
        run(llm, [filler])
        probe.wait_for_copies()
        used += 1
    pytest.fail(
        f"{_MAX_FILLERS} fillers left sessions on the GPU: "
        f"{[str(probe.residency(s.sequence)) for s in sessions]}"
    )


def _late_admissions(changed: dict[str, int]) -> int:
    return sum(
        changed.get(f"restore.{k}", 0)
        for k in ("admit_steps_1", "admit_steps_2_4", "admit_steps_over_4")
    )


@pytest.mark.parametrize(
    ("family", "tp"),
    [("gdn_fat", 1), ("gdn_fat", 2), ("gdn_mtp", 1)],
    ids=lambda v: f"tp{v}" if isinstance(v, int) else v,
)
def test_restore_admits_on_landing(engines, family, tp, record_property):
    spec = EngineSpec(
        family=family, gpu_blocks=96, ram_blocks=384, admission="pack", tp=tp
    )
    engine = engines.get(spec)
    probe, llm = engine.probe, engine.llm
    sessions = _sessions(probe, spec, 6, 8, seed=300)
    for session, result in zip(sessions, run(llm, [s.prompt for s in sessions])):
        session.record(result)
    record_property("fillers", _push_to_ram(llm, probe, spec, sessions))

    before = probe.counters()
    # Where each next turn may resume, at the shallowest: a resume unit short
    # of its session's last prompt end (as `drive` measures turns).
    unit = probe.resume_unit
    floors = [len(s.prompt) - len(s.prompt) % unit - unit for s in sessions]
    # A small next turn each, so one step's token budget admits them all.
    prompts = [s.next_tool_turn(1 / 16) for s in sessions]
    params = greedy(max_tokens=4, logprobs=5)
    resumed = run(llm, prompts, params)
    changed = delta(before, probe.counters())
    for name, value in sorted(changed.items()):
        record_property(name, value)
    save_result("pack restore_admits_on_landing", f"{family}-tp{tp}", changed)

    assert changed.get("restore.completed", 0) >= 1, changed
    assert _late_admissions(changed) == 0, changed
    assert changed.get("restore.extra_short", 0) == 0, changed
    assert changed.get("restore.pin_timeouts", 0) == 0, changed
    assert changed.get("spill.unspilled_evicted_blocks", 0) == 0, changed
    for floor, result in zip(floors, resumed):
        assert result.num_cached_tokens >= floor, (
            result.num_cached_tokens,
            floor,
            probe.residency(result.prompt),
        )
    if spec.model.weights == "random":
        for cold, cached in zip(run(llm, prompts, params, cold=True), resumed):
            assert_outputs_close(cold, cached)


@pytest.mark.parametrize("tp", [1], ids=lambda tp: f"tp{tp}")
def test_whole_restores_start_together(engines, tp, record_property):
    spec = EngineSpec(
        family="gdn_fat", gpu_blocks=128, ram_blocks=384, admission="pack", tp=tp
    )
    engine = engines.get(spec)
    probe, llm = engine.probe, engine.llm
    sessions = _sessions(probe, spec, 4, 4, seed=400)
    for session, result in zip(sessions, run(llm, [s.prompt for s in sessions])):
        session.record(result)
    record_property("fillers", _push_to_ram(llm, probe, spec, sessions))

    before = probe.counters()
    prompts = [s.next_tool_turn(1 / 16) for s in sessions]
    params = greedy(max_tokens=4, logprobs=5)
    resumed = run(llm, prompts, params)
    changed = delta(before, probe.counters())
    for name, value in sorted(changed.items()):
        record_property(name, value)
    stats = probe.tier.restore_stats
    record_property("starts_per_step_max", stats.starts_per_step_max)
    save_result("pack whole_restores_start_together", f"tp{tp}", changed)

    assert changed.get("restore.started", 0) >= 2, changed
    assert changed.get("restore.multi_start_steps", 0) >= 1, changed
    assert changed.get("spill.unspilled_evicted_blocks", 0) == 0, changed
    for cold, cached in zip(run(llm, prompts, params, cold=True), resumed):
        assert_outputs_close(cold, cached)


@pytest.mark.parametrize("admission", ["pack", "legacy"])
@pytest.mark.parametrize("tp", [1], ids=lambda tp: f"tp{tp}")
def test_claimant_not_starved(engines, tp, admission, record_property):
    """One 32-block session that pauses 4 s between turns, among seven small
    agents with none: while it pauses the GPU drops it, and each turn
    restores it in chunks while the others keep arriving. Pack admission must
    get it in quickly; legacy is recorded beside it."""
    spec = EngineSpec(
        family="gdn_fat",
        gpu_blocks=96,
        ram_blocks=640,
        restore_chunk_blocks=16,
        admission=admission,
        tp=tp,
    )
    engine = engines.get(spec)
    probe, llm = engine.probe, engine.llm
    sessions = _sessions(probe, spec, 1, 32, seed=600) + _sessions(
        probe, spec, 7, 4, seed=610
    )
    before = probe.counters()
    turns = drive(
        llm,
        sessions,
        seconds=30.0,
        turn_blocks=1,
        resume_unit=probe.resume_unit,
        max_blocks=48,
        think_s={0: (4.0, 4.0)},
    )
    changed = delta(before, probe.counters())
    for name, value in sorted(changed.items()):
        record_property(name, value)
    long_turns = [t for t in turns if t.session == 0 and t.floor > 0]
    ttfts = [round(t.ttft, 3) for t in long_turns]
    record_property("turns", len(turns))
    record_property("long_session_ttft_s", ttfts)
    print(f"claimant {admission}: turns {len(turns)}, long-session TTFTs {ttfts}")
    save_result(
        "pack claimant_not_starved",
        admission,
        {"turns": len(turns), "long_session_ttft_s": ttfts, "counters": changed},
    )

    assert long_turns, "the long session never came back"
    assert changed.get("restore.gave_up", 0) == 0, changed
    assert changed.get("spill.unspilled_evicted_blocks", 0) == 0, changed
    assert not [t for t in long_turns if t.short_by > 0], [
        (t.floor, t.result.num_cached_tokens) for t in long_turns
    ]
    if admission == "pack":
        assert changed.get("restore.chunked_aborted", 0) == 0, changed
        assert max(t.ttft for t in long_turns) < 15.0, ttfts
