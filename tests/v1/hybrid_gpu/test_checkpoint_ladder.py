# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A session RAM evicts the tail of resumes at a rung of its checkpoint
ladder, not from the start.

RAM evicts a session's tail first, down to its next point to resume at. Its
points used to sit only near its end (exact end, prompt tail, loop start),
so losing them lost it whole. (GPU window 5, 2026-11-23, re-read nine
sessions of 56-160k tokens whole and was first read that way; at least three
of them RAM still held whole, and their restores gave up waiting for room.)
A ladder keeps
a state per doubling stretch back from the end, including inside a long
first prefill, which production's sparse Mamba retention would not cache
otherwise. Here unrelated prompts age a long session out of RAM until its
last page is gone: with the ladder its next turn resumes at a rung and
matches a cold run; without it, from the start. However RAM values its rungs
(`ram_ladder_ranking`): a session about to lose its tail keeps a rung to
resume at."""

import pytest

from .checks import assert_outputs_close
from .engine import EngineSpec
from .workloads import Session, fillers, greedy, run

_SESSION_BLOCKS = 40
_MAX_FILLERS = 60
_RANKINGS = ("fixed", "safe", "expected")


def _spec(tp: int, ladder: int, ranking: str = "fixed") -> EngineSpec:
    return EngineSpec(
        family="gdn_fat",
        connector="tiered",
        ram_eviction="gdsf",
        gpu_blocks=64,
        ram_blocks=64,
        ram_ladder_blocks=ladder,
        ram_ladder_ranking=ranking,
        abandon_gap_multiplier=0.0,
        tp=tp,
    )


def _lose_tail(engines, spec: EngineSpec, record_property, arm: str):
    engine = engines.get(spec)
    probe, llm = engine.probe, engine.llm
    vocab = spec.model.vocab_size
    session = Session.start(
        seed=31,
        vocab_size=vocab,
        block_size=probe.block_size,
        num_blocks=_SESSION_BLOCKS,
        unit=probe.hash_block_size,
    )
    first = run(llm, [session.prompt])[0]
    session.record(first)
    fa = probe.full_attention_group_ids[0]
    used = 0
    for filler in fillers(31, _MAX_FILLERS, 4, vocab, probe.block_size):
        run(llm, [filler])
        probe.wait_for_copies()
        used += 1
        ram = probe.block_ids(session.prompt, fa, "ram")
        gpu = probe.block_ids(session.prompt, fa, "gpu")
        if ram[-1] is None and gpu[-1] is None:
            break
    else:
        pytest.fail(
            f"{arm}: the session kept its tail: {probe.residency(session.prompt)}"
        )
    record_property(f"{arm}.fillers", used)
    record_property(f"{arm}.residency", str(probe.residency(session.prompt)))
    nxt = session.next_tool_turn(0.5)
    second = run(llm, [nxt], greedy(max_tokens=4, logprobs=5))[0]
    record_property(f"{arm}.resumed_at", second.num_cached_tokens)
    if spec.model.weights == "random":
        cold = run(llm, [nxt], greedy(max_tokens=4, logprobs=5), cold=True)[0]
        assert_outputs_close(cold, second)
    return first, second, probe


@pytest.mark.parametrize("tp", [1, 2], ids=lambda tp: f"tp{tp}")
def test_a_session_resumes_at_a_rung_once_its_tail_is_gone(
    engines, tp, record_property
):
    _, flat, _ = _lose_tail(engines, _spec(tp, 0), record_property, "flat")
    for ranking in _RANKINGS:
        first, laddered, _ = _lose_tail(
            engines, _spec(tp, 8, ranking), record_property, f"ladder_{ranking}"
        )
        end = len(first.prompt)
        # Its tail is gone, but a rung below it resumes it: at least the first
        # rung's stretch of the session comes back, at most up to its last page.
        assert 0 < laddered.num_cached_tokens < end, (ranking, laddered)
        assert laddered.num_cached_tokens > flat.num_cached_tokens, (
            ranking,
            flat.num_cached_tokens,
            laddered.num_cached_tokens,
        )
