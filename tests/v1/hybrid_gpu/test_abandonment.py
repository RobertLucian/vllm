# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RAM gives up a session abandoned by its own rhythm before a slow agent's.

RAM GDSF halved a session's resume count every ten minutes, so a slow agent
with long but steady gaps between turns looked cold: 7 of the 9 sessions GPU
window 5 (2026-11-23) lost whole were slow agents. Now a session with turn
gaps on record keeps its count while it is not overdue, and one idle longer
than its patience (a multiple of its longer gap) goes before every live
session. Here a slow agent (gap ~6 s) and a fast one (gap ~0.5 s, resumed
more often, used last) sit in RAM when unrelated prompts force evictions.
The fast one stopped: idle well past its patience, it must lose its blocks
first; with the rule off, it is not the first to go."""

import time

import pytest

from .engine import EngineSpec
from .workloads import Session, fillers, run

_MAX_FILLERS = 60
_SLOW_GAP_S = 6.0
_FAST_GAP_S = 0.5


def _spec(tp: int, multiplier: float) -> EngineSpec:
    return EngineSpec(
        family="gdn_fat",
        connector="tiered",
        ram_eviction="gdsf",
        gpu_blocks=64,
        ram_blocks=48,
        abandon_gap_multiplier=multiplier,
        abandon_floor_s=_FAST_GAP_S,
        tp=tp,
    )


def _turns(llm, session: Session, count: int, gap_s: float) -> None:
    for turn in range(count):
        if turn:
            time.sleep(gap_s)
            prompt = session.next_tool_turn(0.5)
        else:
            prompt = session.prompt
        session.record(run(llm, [prompt])[0])


def _first_lost(engines, spec: EngineSpec, record_property, arm: str) -> str:
    engine = engines.get(spec)
    probe, llm = engine.probe, engine.llm
    vocab = spec.model.vocab_size

    def start(seed: int) -> Session:
        return Session.start(
            seed=seed,
            vocab_size=vocab,
            block_size=probe.block_size,
            num_blocks=8,
            unit=probe.hash_block_size,
        )

    slow, fast = start(41), start(42)
    _turns(llm, slow, 2, _SLOW_GAP_S)
    _turns(llm, fast, 3, _FAST_GAP_S)
    # The fast agent stops: past its patience (2.5 x 0.5 s), well inside the
    # slow one's (2.5 x 6 s).
    time.sleep(4 * _FAST_GAP_S)
    fa = probe.full_attention_group_ids[0]

    def lost(session: Session) -> bool:
        # Its last prompt: computed, so every page was cached.
        ram = probe.block_ids(session.prompt, fa, "ram")
        gpu = probe.block_ids(session.prompt, fa, "gpu")
        return any(r is None and g is None for r, g in zip(ram, gpu))

    started = time.monotonic()
    for used, filler in enumerate(
        fillers(41, _MAX_FILLERS, 4, vocab, probe.block_size)
    ):
        run(llm, [filler])
        probe.wait_for_copies()
        which = [name for name, s in (("slow", slow), ("fast", fast)) if lost(s)]
        if which:
            record_property(f"{arm}.fillers", used + 1)
            record_property(f"{arm}.lost", "/".join(which))
            record_property(f"{arm}.seconds", round(time.monotonic() - started, 1))
            # Both: no order to tell.
            return which[0] if len(which) == 1 else "both"
    record_property(f"{arm}.lost", "neither")
    return "neither"


@pytest.mark.parametrize("tp", [1], ids=lambda tp: f"tp{tp}")
def test_an_overdue_session_goes_before_a_slow_live_one(engines, tp, record_property):
    assert _first_lost(engines, _spec(tp, 2.5), record_property, "rule") == "fast"
    # Without the rule the fast agent, resumed more often and used last, is
    # never the first to go (the fillers, worth less, may outlast both).
    assert _first_lost(engines, _spec(tp, 0.0), record_property, "off") != "fast"
