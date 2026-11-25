# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Head-of-line blocking at equal pools: an experiment, measured, not judged.

On production's pi trace the tiered connector's cheap resumes waited minutes at
equal pools where upstream's eager mirror served them in seconds (2026-11-23).
Under legacy admission:
- `_behind_head` holds every request queued after a restore that waits for
  room, including sessions wholly on the GPU;
- a restore loads its session but not room for the tokens past its point,
  so the request then waits again, and the scheduler's waiting loop ends
  there.

Pack admission lets every request that fits go, starts restores side by side
with room for their first chunk past the point (the scheduler's own count),
and has the earliest request that does not fit claim the room it waits for
once it has waited `claim_after_s`. A bounded search looks only as deep as the
restore gain it checks; the whole walk is today's.

Each setting runs its arms one after another on one GPU: the eager mirror
(upstream's design), then the tiered connector's arms, twice, the rounds
interleaved. Settings:
- window H's workload (8 agents, one token a turn);
- the same with production's scheduler flags (prompts read every 6th step,
  a chunk cap, priority, async scheduling);
- a production-like one: 16 agents with pauses between turns, sessions up to
  48 blocks that restore in chunks, and RAM sized to 0.9 of the working set
  a spare-RAM arm measures.

A watch reads the scheduler and the connector after every step and changes
nothing: why each waiting request waits (the connector's reason this step),
how long steps take, and its own time. Restored turns are those the connector
admitted from a restore (`recent_resumes`). Only safety is asserted, and that
each arm's switches took effect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from vllm.v1.simple_kv_offload.tiered import WAIT_REASONS

from .checks import delta, save_result
from .engine import EngineSpec
from .probes import Probe
from .workloads import Session, Turn, drive

_FAMILY = "gdn_fat"
_SECONDS = 30.0
# Window H's workload (test_eager_vs_lazy.py): 8 agents, 96 GPU + 384 RAM.
_A_GPU, _A_AGENTS, _A_RAM = 96, 8, 384
# Production's shape: the GPU holds about five of the sixteen sessions,
# sessions restore in several chunks, agents pause between turns, and RAM is
# over-committed (sized from the working set the spare-RAM arm measures).
# Pinned RAM is about 17 MiB a block: the spare arm's 640 stay well inside
# the test pod's 40 GiB beside the other TP=1 test.
_P_GPU, _P_AGENTS, _P_SPARE_RAM = 192, 16, 640
_P_BLOCKS = dict(start_blocks=8, turn_blocks=2, max_blocks=48)
_P_THINK_S = (0.25, 1.0)
_OVERCOMMIT = 0.9
# Production's scheduler flags (talos/manifests/llms/04-llama-server.yaml):
# prompts read every 6th step, 4096 batched tokens, a chunk cap (production's
# is one 1,600-token block; about two of this model's), priority scheduling
# with agents at priority 10, async scheduling.
_PRODUCTION_FLAGS = dict(
    prefill_schedule_interval=6,
    max_num_batched_tokens=4096,
    long_prefill_token_threshold=1024,
    priority_policy=True,
    async_scheduling=True,
)
_AGENT_PRIORITY = 10
_WAITS = ("restore_loading", "restore_done", *WAIT_REASONS, "other")


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def _ms(seconds: float | None) -> int | None:
    return None if seconds is None else round(1000 * seconds)


@dataclass
class _Watch:
    """Per step, why each waiting request waits and how long the step took;
    for the eager mirror, which requests loaded from RAM."""

    probe: Probe
    sessions: list[Session]
    # Steps between samples of the sessions' working set (0: none).
    sample_every: int = 0
    seconds: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(_WAITS, 0.0)
    )
    steps: list[float] = field(default_factory=list)
    loaded: set[str] = field(default_factory=set)
    working_set: int = 0
    # The watch's own time.
    watch_s: float = 0.0

    def on_step(self, started: float) -> None:
        took = time.monotonic() - started
        begun = time.monotonic()
        self.steps.append(took)
        scheduler = self.probe.scheduler
        tier = self.probe.tier
        restores = getattr(tier, "_restores", None)
        reasons = getattr(tier, "_last_wait", {})
        for queue in (scheduler.waiting, getattr(scheduler, "kv_holding_waiting", ())):
            for request in queue:
                self.seconds[self._why(request.request_id, restores, reasons)] += took
        if restores is None:
            self.loaded.update(getattr(tier, "_reqs_to_load", ()))
        if self.sample_every and len(self.steps) % self.sample_every == 0:
            self.working_set = max(self.working_set, self._working_set())
        self.watch_s += time.monotonic() - begun

    @staticmethod
    def _why(
        request_id: str, restores: dict | None, reasons: dict[str, str]
    ) -> str:
        if restores is None:
            return "other"
        restore = restores.get(request_id)
        if restore is not None:
            return "restore_done" if restore.done else "restore_loading"
        # No reason: the scheduler allocated nothing for it and was not
        # refused (it did not reach it, or a budget or cap stopped it).
        return reasons.get(request_id, "scheduler")

    def _working_set(self) -> int:
        """Blocks either tier caches of the sessions' latest sequences:
        attention pages per block, recurrent states per resume point."""
        probe = self.probe
        total = 0
        for session in self.sessions:
            if not session.sequence:
                continue
            marks = probe.residency(session.sequence)
            for group_id in range(len(probe.group_block_sizes)):
                if group_id in probe.mamba_group_ids:
                    total += sum(m != "." for m in marks.marks[group_id])
                else:
                    total += sum(m != "." for m in marks.blocks(group_id))
        return total


def _run(
    engines,
    spec: EngineSpec,
    agents: int,
    max_tokens: int,
    sample_every: int = 0,
    start_blocks: float = 4,
    turn_blocks: float = 1,
    max_blocks: float = 24,
    think_s: tuple[float, float] | None = None,
    priority: int = 0,
    seconds: float = _SECONDS,
) -> dict[str, Any]:
    engine = engines.get(spec)
    probe = engine.probe
    sessions = [
        Session.start(
            seed=200 + i,
            vocab_size=spec.model.vocab_size,
            block_size=probe.block_size,
            num_blocks=start_blocks,
            unit=probe.hash_block_size,
        )
        for i in range(agents)
    ]
    watch = _Watch(probe, sessions, sample_every)
    before = probe.counters()
    turns = drive(
        engine.llm,
        sessions,
        seconds=seconds,
        turn_blocks=turn_blocks,
        resume_unit=probe.resume_unit,
        user_every=5,
        max_blocks=max_blocks,
        restart_blocks=start_blocks,
        max_tokens=max_tokens,
        on_step=watch.on_step,
        think_s=think_s,
        priority=priority,
    )
    return _summary(
        turns,
        watch,
        delta(before, probe.counters()),
        probe.peaks(),
        getattr(probe.tier, "recent_resumes", None),
    )


def _summary(
    turns: list[Turn],
    watch: _Watch,
    counters: dict[str, int],
    peaks: dict[str, int],
    resumes: dict[str, tuple[str, int]] | None,
) -> dict:
    # The scheduler's ids are the turns' with a random suffix (`InputProcessor`).
    by_turn = {k.rsplit("-", 1)[0]: v for k, v in (resumes or {}).items()}
    loaded = {k.rsplit("-", 1)[0] for k in watch.loaded}

    def tier_of(turn: Turn) -> str:
        if resumes is None:
            return "ram" if turn.request_id in loaded else "gpu"
        entry = by_turn.get(turn.request_id)
        return "unknown" if entry is None else entry[0]

    resumed = [t for t in turns if t.floor > 0]
    restored = [t for t in resumed if tier_of(t) == "ram"]
    resident = [t for t in resumed if tier_of(t) == "gpu"]
    short = [t for t in turns if t.short_by > 0]
    steps = watch.steps
    stepping_ns = max(1, int(1e9 * sum(steps)))

    def get(name: str) -> int:
        return counters.get(name, 0)

    parts = {
        part: get(f"restore.{part}_ns")
        for part in (
            "lookup",
            "lookup_search",
            "lookup_junction",
            "lookup_room",
            "estimate",
            "observe_queue",
            "step",
            "stats",
        )
    }
    parts["spill_scan"] = get("pool.spill_scan_ns")
    parts["spill_reorder"] = get("pool.spill_reorder_ns")
    connector_ns = sum(parts[p] for p in ("lookup", "observe_queue", "step", "stats"))
    calls = get("restore.search_calls")
    return {
        "turns": len(turns),
        "short_resumes": len(short),
        "recomputed_tokens": sum(t.short_by for t in short),
        "ttft_p50_ms": _ms(_pct([t.ttft for t in resumed], 50)),
        "ttft_p95_ms": _ms(_pct([t.ttft for t in resumed], 95)),
        "resident_turns": len(resident),
        "resident_ttft_p50_ms": _ms(_pct([t.ttft for t in resident], 50)),
        "resident_ttft_p95_ms": _ms(_pct([t.ttft for t in resident], 95)),
        "restored_turns": len(restored),
        "restored_ttft_p50_ms": _ms(_pct([t.ttft for t in restored], 50)),
        "restored_ttft_p95_ms": _ms(_pct([t.ttft for t in restored], 95)),
        "unclassified_turns": len(resumed) - len(resident) - len(restored),
        "decode_p50_ms": _ms(_pct([t.finished - t.first_token for t in turns], 50)),
        "decode_p95_ms": _ms(_pct([t.finished - t.first_token for t in turns], 95)),
        "steps": len(steps),
        "stepping_s": round(sum(steps), 2),
        "step_p50_ms": _ms(_pct(steps, 50)),
        "step_p95_ms": _ms(_pct(steps, 95)),
        "step_max_ms": _ms(max(steps, default=None)),
        "watch_s": round(watch.watch_s, 2),
        # Connector time as a share of the time steps took, and by part in ms
        # per second of stepping.
        "connector_pct": round(100 * connector_ns / stepping_ns, 2),
        "ms_per_s": {
            part: round(1e3 * ns / stepping_ns, 2) for part, ns in parts.items()
        },
        "search_points_per_call": round(get("restore.search_points") / calls, 1)
        if calls
        else None,
        "search_units_per_call": round(get("restore.search_units") / calls, 1)
        if calls
        else None,
        "search_mismatches": get("restore.search_mismatches"),
        "spill_scan_blocks_per_call": round(
            get("pool.spill_scan_blocks") / max(1, get("pool.spill_scan_calls")), 1
        ),
        "room": {
            name: get(f"restore.{name}")
            for name in (
                "extra_exact_admissions",
                "extra_short",
                "extra_short_blocks",
                "extra_unused_blocks",
                "extra_released_unadmitted",
                "extra_regrown",
                "extra_shrunk",
                "extra_zero_chunk",
                "admitted_hit_short",
                "admitted_hit_past",
            )
        },
        "admit_steps": {
            k: get(f"restore.admit_steps_{k}") for k in ("0", "1", "2_4", "over_4")
        },
        "lookups": {
            k.removeprefix("lookup."): v
            for k, v in counters.items()
            if k.startswith("lookup.")
        },
        "landed_waits": {
            k.removeprefix("landed."): v
            for k, v in counters.items()
            if k.startswith("landed.")
        },
        "peaks": peaks,
        "wait_s": {k: round(v, 1) for k, v in watch.seconds.items() if v},
        "working_set_blocks": watch.working_set,
        "counters": {
            k: v
            for k, v in counters.items()
            if not k.startswith(("prune.", "lookup.", "landed."))
            and not k.endswith("_max")
        },
    }


def _equal(connector: str, tp: int, gpu: int, ram: int, **kwargs: Any) -> EngineSpec:
    """An arm whose RAM may hold what the GPU holds: it gets both of the
    tiered arm's pools as RAM (equal pools)."""
    return EngineSpec(
        family=_FAMILY,
        connector=connector,  # type: ignore[arg-type]
        gpu_blocks=gpu,
        ram_blocks=gpu + ram,
        tp=tp,
        **kwargs,
    )


def _tiered(tp: int, gpu: int, ram: int, **kwargs: Any) -> EngineSpec:
    return EngineSpec(
        family=_FAMILY,
        connector="tiered",
        gpu_blocks=gpu,
        ram_blocks=ram,
        tp=tp,
        **kwargs,
    )


def _report(name: str, results: dict[str, dict], record_property) -> None:
    for arm, summary in results.items():
        for key, value in summary.items():
            record_property(f"{arm}.{key}", value)
        print(f"head-of-line {name} {arm}: {summary}")
        save_result(f"head-of-line {name}", arm, summary)


def _assert_safe(results: dict[str, dict], arms: tuple[str, ...]) -> None:
    for arm in arms:
        summary = results[arm]
        assert summary["counters"].get("spill.unspilled_evicted_blocks", 0) == 0, (
            arm,
            summary,
        )


def _assert_switches(results: dict[str, dict], arms: dict[str, EngineSpec]) -> None:
    """Each arm ran what its spec says: legacy keeps no room for a first
    chunk; pack does, for every restore it admitted; a bounded search
    visits fewer points per call than the whole walk under the same
    admission."""
    points: dict[tuple[str, bool], list[float]] = {}
    for arm, spec in arms.items():
        if spec.connector != "tiered":
            continue
        summary = results[arm]
        room = summary["room"]
        kept = sum(
            room[k]
            for k in ("extra_exact_admissions", "extra_short", "extra_unused_blocks")
        )
        if spec.admission == "legacy":
            assert kept == 0, (arm, room)
        elif summary["counters"].get("restore.completed", 0):
            assert kept > 0, (arm, room)
        assert summary["search_mismatches"] == 0, (arm, summary)
        if summary["search_points_per_call"] is not None:
            key = (spec.admission, spec.bounded_search)
            points.setdefault(key, []).append(summary["search_points_per_call"])
    for (admission, bounded), values in points.items():
        whole = points.get((admission, False))
        if bounded and whole:
            assert sum(values) / len(values) < sum(whole) / len(whole), points


def _window_h_arms(tp: int, **flags: Any) -> dict[str, EngineSpec]:
    """Today's code (legacy, whole walk), legacy with the bounded search,
    pack with the whole walk, pack (bounded): each twice, interleaved."""
    tiered = {
        "legacy": dict(admission="legacy", bounded_search=False),
        "legacy_bounded": dict(admission="legacy"),
        "pack_whole": dict(admission="pack", bounded_search=False),
        "pack": dict(admission="pack"),
    }
    arms = {"eager": _equal("eager", tp, _A_GPU, _A_RAM, **flags)}
    for round_ in (1, 2):
        for name, kwargs in tiered.items():
            arms[f"{name}_{round_}"] = _tiered(tp, _A_GPU, _A_RAM, **kwargs, **flags)
    return arms


@pytest.mark.parametrize("tp", [1, 2], ids=lambda v: f"tp{v}")
def test_head_of_line_window_h(engines, tp, record_property):
    """Window H's workload, one token a turn. Window S (2026-11-24, TP 1 /
    TP 2): eager 477 / 487 turns; legacy 148, 239 / 365, 366; pack with the
    old room estimate (7 blocks over per restore) 366, 364 / 326, 334."""
    arms = _window_h_arms(tp)
    results = {
        arm: _run(engines, spec, _A_AGENTS, max_tokens=1) for arm, spec in arms.items()
    }
    _report(f"window-h tp{tp}", results, record_property)
    lazy = tuple(arm for arm in arms if arm != "eager")
    _assert_safe(results, lazy)
    for arm in lazy:
        assert results[arm]["short_resumes"] == 0, (arm, results[arm])
    _assert_switches(results, arms)


@pytest.mark.parametrize("tp", [2], ids=lambda v: f"tp{v}")
def test_head_of_line_production_flags(engines, tp, record_property):
    """Window H's workload under production's scheduler flags, one round:
    today's code with the whole walk and with the bounded search, and pack.
    Window T (2026-11-24): legacy 342 turns (its lookups 12% of step time),
    pack 329."""
    arms = {
        "legacy": _tiered(
            tp,
            _A_GPU,
            _A_RAM,
            admission="legacy",
            bounded_search=False,
            **_PRODUCTION_FLAGS,
        ),
        "legacy_bounded": _tiered(
            tp, _A_GPU, _A_RAM, admission="legacy", **_PRODUCTION_FLAGS
        ),
        "pack": _tiered(tp, _A_GPU, _A_RAM, admission="pack", **_PRODUCTION_FLAGS),
    }
    results = {
        arm: _run(engines, spec, _A_AGENTS, max_tokens=1, priority=_AGENT_PRIORITY)
        for arm, spec in arms.items()
    }
    _report(f"production-flags tp{tp}", results, record_property)
    _assert_safe(results, tuple(arms))
    for arm in arms:
        assert results[arm]["short_resumes"] == 0, (arm, results[arm])
    _assert_switches(results, arms)


@pytest.mark.parametrize("tp", [1, 2], ids=lambda v: f"tp{v}")
def test_head_of_line_production_like(engines, tp, record_property):
    """Production's shape, 16 tokens a turn. The spare-RAM arm runs first and
    measures the sessions' working set; the other arms get pools of
    `_OVERCOMMIT` of it."""
    workload = dict(**_P_BLOCKS, think_s=_P_THINK_S)
    common = dict(max_num_seqs=16)
    spare = _run(
        engines,
        _tiered(tp, _P_GPU, _P_SPARE_RAM, **common),
        _P_AGENTS,
        max_tokens=16,
        sample_every=50,
        **workload,
    )
    ram = max(32, int(_OVERCOMMIT * spare["working_set_blocks"]) - _P_GPU)
    arms = {
        "eager": _equal("eager", tp, _P_GPU, ram, **common),
        "legacy": _tiered(tp, _P_GPU, ram, admission="legacy", **common),
        "pack": _tiered(tp, _P_GPU, ram, admission="pack", **common),
    }
    results = {"spare": spare}
    for arm, spec in arms.items():
        results[arm] = _run(engines, spec, _P_AGENTS, max_tokens=16, **workload)
    record_property("ram_blocks", ram)
    _report(f"production-like tp{tp} ram={ram}", results, record_property)
    _assert_safe(results, ("spare", "legacy", "pack"))
