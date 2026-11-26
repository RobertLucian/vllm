# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RAM GDSF ranks a session's checkpoint-ladder rungs with its pages
(`ram_ladder_ranking`), and the connector counts what each tier holds.

RAM cuts a session's segments deepest first, and a segment worth less per
block than every one above it goes right after them: the point on top of it
is never resumed at. At production's depth (sessions ~150k tokens, the
prefill cost per token growing with it) that is every rung of a one-turn
session: its far segments are worth less per block than its near ones. With
"safe" such rungs go at once, from both tiers, as long as the points the
session can be resumed at stay the same; "fixed" keeps them until the tail
above them goes (window 7, 2026-11-24, held up to ~160 of 1,329 blocks that
way, an estimate). The tests here run at production's depth by scaling the
cost model's depth term to the tests' session lengths (`scaled_cost`).
- One long session, "fixed" against "safe": the rungs the cost model says no
  cut stops at leave both tiers in "safe" only; the session then holds its
  tail through at least as many unrelated prompts, and its next turn matches
  a cold run.
- The tier counts add up: every block of each pool once, matching a tally of
  the pools, through Prometheus and the summary log line.
- A busy run in production's shape (16 agents with pauses, sessions up to 48
  blocks, RAM 0.9 of the working set): "fixed", "safe" and "expected", two
  rounds each, measured, not judged; each arm's switch took effect and no
  block left uncopied.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest
from prometheus_client import REGISTRY

from vllm.v1.core.hybrid_checkpoint.cost import PrefillCostModel
from vllm.v1.simple_kv_offload import tiered

from .checks import assert_outputs_close, delta, save_result
from .engine import EngineSpec, scaled_cost
from .probes import Probe
from .test_head_of_line import (
    _A_AGENTS,
    _A_GPU,
    _A_RAM,
    _OVERCOMMIT,
    _P_AGENTS,
    _P_BLOCKS,
    _P_GPU,
    _P_SPARE_RAM,
    _P_THINK_S,
    _assert_safe,
    _report,
    _summary,
    _tiered,
    _Watch,
)
from .workloads import Session, drive, fillers, greedy, run

# gdn_fat's block in tokens (window U, 2026-11-24: an 8-block session's
# boundary point at 4,592 = 8 x 576 - 16).
_BLOCK = 576
_SESSION_BLOCKS = 40
_COST = scaled_cost(_SESSION_BLOCKS * _BLOCK)
_MAX_FILLERS = 60
# Fillers past the one that took the session's last page: the cuts under it
# go at the next evictions, below the clock.
_CASCADE_FILLERS = 3
# A busy arm runs long enough for the clock's rise to be measured (30 s, for
# "expected").
_BUSY_S = 45.0
_SUMMARY_S = 5.0


def _spec(tp: int, ranking: str, **kwargs: Any) -> EngineSpec:
    return EngineSpec(
        family="gdn_fat",
        connector="tiered",
        ram_eviction="gdsf",
        gpu_blocks=64,
        ram_blocks=64,
        ram_ladder_ranking=ranking,  # type: ignore[arg-type]
        abandon_gap_multiplier=0.0,
        prefill_cost=_COST,
        tp=tp,
        **kwargs,
    )


def _predicted_drops(
    rungs: list[int], reach: int, cost: tuple[float, float], states: int
) -> set[int]:
    """The rungs (token positions) no cut could stop at, by the cost model:
    a one-turn session's points are its rungs and its end, each segment its
    pages plus the states at its top, valued per block; a point is resumed
    at only if the segment under it is worth more per block than every one
    above it. Dropped deepest first while the points that can be resumed at
    stay the same."""
    model = PrefillCostModel(*cost)

    def resumable(points: list[int]) -> set[int]:
        values = []
        lower = 0
        for top in points:
            blocks = (top - lower) // _BLOCK + states
            values.append(model.seconds(lower, top) / blocks)
            lower = top
        out, above = set(), float("-inf")
        for k in range(len(points) - 1, 0, -1):
            above = max(above, values[k])
            if values[k - 1] > above + 1e-12:
                out.add(points[k - 1])
        return out

    points = sorted(rungs) + [reach]
    keep = resumable(points)
    dropped = set()
    for rung in sorted(rungs, reverse=True):
        if rung in keep:
            continue
        trial = [p for p in points if p != rung]
        if resumable(trial) == keep:
            points = trial
            dropped.add(rung)
    return dropped


def _rungs(probe: Probe) -> tuple[set[int], dict[int, str]]:
    """The positions of the ladder rungs the connector keeps, and of those
    that went (-> why), over every session (only the long session here has
    a ladder)."""
    tier = probe.tier
    kept = {p for rungs in tier._rungs_by.values() for p in rungs.values()}
    gone = {p: why for lost in tier._dropped_rungs.values() for p, why in lost}
    return kept, gone


def _states_held(probe: Probe, tokens: list[int], position: int) -> dict[str, bool]:
    """Whether each tier holds every Mamba group's state at `position`."""
    unit = position // probe.hash_block_size - 1
    return {
        tier: all(
            unit in probe.cached_units(tokens, group, tier)
            for group in probe.mamba_group_ids
        )
        for tier in ("gpu", "ram")
    }


def _one_session(engines, spec: EngineSpec, record_property, arm: str) -> dict:
    engine = engines.get(spec)
    probe, llm = engine.probe, engine.llm
    record_property(f"{arm}.block_size", probe.block_size)
    vocab = spec.model.vocab_size
    session = Session.start(
        seed=61,
        vocab_size=vocab,
        block_size=probe.block_size,
        num_blocks=_SESSION_BLOCKS,
        unit=probe.hash_block_size,
    )
    before = probe.counters()
    session.record(run(llm, [session.prompt])[0])
    probe.wait_for_copies()
    # A step drops what the policy chose (`_drain_dropped_points`) and the
    # GPU prunes it.
    run(llm, fillers(62, 1, 1, vocab, probe.block_size))
    probe.wait_for_copies()
    kept, gone = _rungs(probe)
    held = {p: _states_held(probe, session.prompt, p) for p in [*kept, *gone]}
    # The session's own drops (its next turns, the cold run's too, recompute
    # and drop rungs of their own).
    dropped_now = delta(before, probe.counters()).get("ladder.dropped_safe", 0)
    fa = probe.full_attention_group_ids[0]
    used = 0
    for filler in fillers(61, _MAX_FILLERS, 4, vocab, probe.block_size):
        run(llm, [filler])
        probe.wait_for_copies()
        used += 1
        ram = probe.block_ids(session.prompt, fa, "ram")
        gpu = probe.block_ids(session.prompt, fa, "gpu")
        if ram[-1] is None and gpu[-1] is None:
            break
    else:
        pytest.fail(f"{arm}: the session kept its tail")
    for filler in fillers(63, _CASCADE_FILLERS, 4, vocab, probe.block_size):
        run(llm, [filler])
        probe.wait_for_copies()
    nxt = session.next_tool_turn(0.5)
    params = greedy(max_tokens=4, logprobs=5)
    resumed = run(llm, [nxt], params)[0]
    cold = run(llm, [nxt], params, cold=True)[0]
    assert_outputs_close(cold, resumed)
    changed = delta(before, probe.counters())
    out = dict(
        rungs=sorted([*kept, *gone]),
        dropped=gone,
        dropped_counted=dropped_now,
        held={p: h for p, h in sorted(held.items())},
        fillers_until_tail_left=used,
        resumed_at=resumed.num_cached_tokens,
        prompt_tokens=len(session.prompt),
        # Blocks a boundary point's states take beside its page (RAM GDSF
        # counts every group other than the pages' as a state).
        state_groups=len(probe.tier.ram_policy.state_group_ids),
        counters={
            k: v for k, v in changed.items() if k.startswith(("ram.", "ladder."))
        },
    )
    for name, value in out.items():
        record_property(f"{arm}.{name}", value)
    save_result("ladder value one session", arm, out)
    return out


@pytest.mark.parametrize("tp", [1, 2], ids=lambda tp: f"tp{tp}")
def test_rungs_no_cut_stops_at_leave_both_tiers(engines, tp, record_property):
    fixed = _one_session(engines, _spec(tp, "fixed"), record_property, "fixed")
    safe = _one_session(engines, _spec(tp, "safe"), record_property, "safe")
    assert fixed["rungs"] == safe["rungs"] and fixed["rungs"], (fixed, safe)
    # "fixed" drops nothing on its own: every rung stays in some tier.
    assert not fixed["dropped"], fixed
    assert all(h["gpu"] or h["ram"] for h in fixed["held"].values()), fixed
    # "safe" drops exactly what the cost model says no cut stops at, from
    # both tiers; the rest stay.
    expected = _predicted_drops(
        safe["rungs"], safe["prompt_tokens"], _COST, safe["state_groups"]
    )
    assert set(safe["dropped"]) == expected, (safe, expected)
    assert set(safe["dropped"].values()) <= {"safe"}, safe
    for position, held in safe["held"].items():
        if position in safe["dropped"]:
            assert not held["gpu"] and not held["ram"], (position, safe)
        else:
            assert held["gpu"] or held["ram"], (position, safe)
    assert safe["dropped_counted"] == len(expected), safe
    # Their blocks hold other data: the session keeps its tail at least as
    # long. Where each resumes once the tail is gone is recorded, not judged:
    # window A (2026-11-25) had "fixed" resume at its 32-block rung (orphans,
    # duplicates and the cut segment's dead pages went before the segment
    # under it) and "safe" from the start.
    assert safe["fillers_until_tail_left"] >= fixed["fillers_until_tail_left"], (
        fixed,
        safe,
    )


def _prometheus_total(name: str) -> float:
    total = 0.0
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name == name:
                total += sample.value
    return total


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.mark.parametrize("tp", [1, 2], ids=lambda tp: f"tp{tp}")
def test_tier_contents_add_up(engines, tp, record_property):
    spec = _tiered(
        tp,
        _A_GPU,
        _A_RAM,
        ram_eviction="gdsf",
        log_stats=True,
        summary_log_s=_SUMMARY_S,
    )
    engine = engines.get(spec)
    probe = engine.probe
    # After the engine starts: its logging setup clears the handlers of
    # vLLM's loggers (`dictConfig`).
    handler = _Lines()
    tiered.logger.addHandler(handler)
    try:
        sessions = [
            Session.start(
                seed=300 + i,
                vocab_size=spec.model.vocab_size,
                block_size=probe.block_size,
                num_blocks=4,
                unit=probe.hash_block_size,
            )
            for i in range(_A_AGENTS)
        ]
        drive(
            engine.llm,
            sessions,
            seconds=15.0,
            turn_blocks=1,
            resume_unit=probe.resume_unit,
            user_every=5,
            max_blocks=24,
        )
    finally:
        tiered.logger.removeHandler(handler)
    counts, duplicates = probe.tier_contents()
    for name, value in counts.items():
        record_property(f"{name[0]}.{name[1]}", value)
    gpu, cpu = probe.gpu_pool, probe.cpu_pool
    for tier, pool in (("gpu", gpu), ("ram", cpu)):
        blocks = [b for b in pool.blocks if not b.is_null]
        kinds = {k: n for (t, k), n in counts.items() if t == tier}
        assert sum(kinds.values()) == len(blocks), (tier, kinds, len(blocks))
        in_use = sum(1 for b in blocks if b.ref_cnt)
        free = sum(1 for b in blocks if not b.ref_cnt and b.block_hash is None)
        assert kinds["in_use"] == in_use, (tier, kinds, in_use)
        assert kinds["free"] == free, (tier, kinds, free)
    cached_gpu = gpu.cached_block_hash_to_block
    both = sum(
        1
        for b in cpu.blocks
        if b.block_hash is not None and cached_gpu.get_one_block(b.block_hash)
    )
    assert sum(duplicates.values()) == both, (duplicates, both)
    assert _prometheus_total("vllm:tiered_kv_tier_blocks") > 0
    tier_lines = [line for line in handler.lines if line.startswith("Tiered KV tiers")]
    record_property("tier_lines", len(tier_lines))
    for line in tier_lines[-2:]:
        print(line)
    assert tier_lines, handler.lines[-20:]


def _busy(engines, spec: EngineSpec, seconds: float, **workload: Any) -> dict:
    """`test_head_of_line._run`, also counting what each tier holds and the
    RAM blocks no eviction ranks at the end (before the engine is reused)."""
    engine = engines.get(spec)
    probe = engine.probe
    sessions = [
        Session.start(
            seed=200 + i,
            vocab_size=spec.model.vocab_size,
            block_size=probe.block_size,
            num_blocks=workload["start_blocks"],
            unit=probe.hash_block_size,
        )
        for i in range(_P_AGENTS)
    ]
    watch = _Watch(probe, sessions, workload.get("sample_every", 0))
    before = probe.counters()
    turns = drive(
        engine.llm,
        sessions,
        seconds=seconds,
        turn_blocks=workload["turn_blocks"],
        resume_unit=probe.resume_unit,
        user_every=5,
        max_blocks=workload["max_blocks"],
        restart_blocks=workload["start_blocks"],
        max_tokens=16,
        on_step=watch.on_step,
        think_s=workload["think_s"],
    )
    summary = _summary(
        turns,
        watch,
        delta(before, probe.counters()),
        probe.peaks(),
        probe.tier.recent_resumes,
    )
    counts, duplicates = probe.tier_contents()
    summary["tier_blocks"] = {f"{t}.{k}": n for (t, k), n in counts.items() if n}
    summary["duplicate_blocks"] = duplicates
    policy = probe.tier.ram_policy
    summary["ram_unranked_blocks"] = (
        None
        if policy is None
        else dict(zip(("page", "state"), policy.unranked_blocks()))
    )
    summary["clock_rate"] = None if policy is None else policy.clock_rate
    summary["ladder_p"] = (
        None
        if policy is None
        else {f"{b}.{o}": n for (b, o), n in sorted(policy.ladder_p.items())}
    )
    return summary


@pytest.mark.parametrize("tp", [1], ids=lambda tp: f"tp{tp}")
def test_ladder_ranking_busy(engines, tp, record_property):
    """Production's shape at production's depth: the spare-RAM arm measures
    the working set, then each ranking runs twice, interleaved, on RAM of
    `_OVERCOMMIT` of it."""
    workload = dict(**_P_BLOCKS, think_s=_P_THINK_S)
    common = dict(max_num_seqs=16, ram_eviction="gdsf")
    # Sessions grow from 8 to 48 blocks: scaled at their middle.
    cost = scaled_cost(28 * _BLOCK)
    started = time.monotonic()
    spare = _busy(
        engines,
        _tiered(tp, _P_GPU, _P_SPARE_RAM, prefill_cost=cost, **common),
        _BUSY_S,
        sample_every=50,
        **workload,
    )
    ram = max(32, int(_OVERCOMMIT * spare["working_set_blocks"]) - _P_GPU)
    arms = {
        f"{ranking}_{round_}": _tiered(
            tp,
            _P_GPU,
            ram,
            ram_ladder_ranking=ranking,
            prefill_cost=cost,
            **common,
        )
        for round_ in (1, 2)
        for ranking in ("fixed", "safe", "expected")
    }
    results = {"spare": spare}
    for arm, spec in arms.items():
        results[arm] = _busy(engines, spec, _BUSY_S, **workload)
    record_property("ram_blocks", ram)
    record_property("seconds", round(time.monotonic() - started))
    _report(f"ladder-ranking tp{tp} ram={ram}", results, record_property)
    save_result("ladder ranking busy", "ram_blocks", {"ram_blocks": ram})
    _assert_safe(results, tuple(results))
    for arm, spec in arms.items():
        counters = results[arm]["counters"]
        dropped = counters.get("ladder.dropped_safe", 0) + counters.get(
            "ladder.dropped_ranked", 0
        )
        if spec.ram_ladder_ranking == "fixed":
            assert dropped == 0 and counters.get("ram.rungs", 0) == 0, (arm, counters)
        else:
            # At production's depth a one-turn session's far rungs are no
            # cut's to stop at.
            assert counters.get("ladder.dropped_safe", 0) > 0, (arm, counters)
