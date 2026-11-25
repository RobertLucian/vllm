# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The room a restore keeps for its request's first chunk is the room its
admission takes (`TieredCPUOffloadScheduler._estimate_extra`, the scheduler's
own count: `Scheduler.first_chunk`).

Sessions pushed to RAM resume one at a time on an idle GPU, so the first
chunk gets a step's whole budget, in shapes that set the chunk differently:
- at an exact end inside a block: the chunk stops at the block's end, or at
  the prompt's end before it;
- at a block boundary (no exact-end states): the chunk runs a step's budget,
  cut to whole blocks, or to a rung of the checkpoint ladder;
- with spec decode (MTP), attention only, a sliding window, and TP 2.
Each shape first checks it happened (a restore, the point it resumed at, the
chunk's cut), then that the room was exact: no block short or unused, the
restore admitted the step its load landed, no Mamba bookkeeping for the
request before its admission, and the outputs of a full prefill. With an
urgent request beside it (the low-priority cap cuts the chunk) the room may
only be too large, never short.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import pytest

from vllm import TokensPrompt
from vllm.v1.core.single_type_kv_cache_manager import MambaManager

from .checks import assert_outputs_close, delta, save_result
from .engine import EngineSpec
from .probes import Probe
from .test_pack_admission import _push_to_ram, _sessions
from .workloads import Result, Session, greedy, run

_ids = itertools.count()
# Attention-only and sliding-window models have 16-token blocks: a small
# budget lets a few blocks' message outrun it.
_SMALL_BUDGET = 64
# Sessions and first turns: a prompt a quarter block past its last boundary
# whose turn generates one token resumes at an exact end inside a block, its
# state from a prefill (from a decode instead, a full prefill's reference
# parted at a near tie at TP 2 in window U); whole blocks resume at a
# boundary where hits land on boundaries.
_MID_BLOCKS, _BOUNDARY_BLOCKS = 8.25, 8
# A resume alone never takes this long; past it the test fails, not hangs.
_RESUME_TIMEOUT_S = 120.0


def _pack(**kwargs) -> EngineSpec:
    """Pack admission (the room estimate is its), 96 GPU + 384 RAM blocks."""
    return EngineSpec(gpu_blocks=96, ram_blocks=384, admission="pack", **kwargs)


@dataclass(frozen=True)
class _Shape:
    name: str
    # The next turn's new tokens, in blocks.
    turn_blocks: float
    # Where the resume lands: inside a block, or on a boundary.
    mid_block: bool
    # The chunk the scheduler cuts: at the step's budget, at a rung.
    budget_cut: bool = False
    rung_cut: bool = False


_MID = (
    _Shape("exact end, short message", 1 / 16, mid_block=True),
    _Shape("exact end, message past the block", 1, mid_block=True),
)
_ALIGNED = (
    _Shape("boundary, short message", 1, mid_block=False),
    _Shape("boundary, message past the budget", 6, mid_block=False, budget_cut=True),
    _Shape("boundary, a rung in the chunk", 10, mid_block=False, rung_cut=True),
)
_ATTENTION = (
    _Shape("boundary, short message", 1, mid_block=False),
    _Shape("boundary, message past the budget", 6, mid_block=False, budget_cut=True),
)

_CASES = {
    "gdn_fat-tp1": (_pack(family="gdn_fat"), _MID),
    "gdn_fat-tp2": (
        _pack(family="gdn_fat", tp=2),
        _MID,
    ),
    # No exact ends, and hits on block boundaries only (with 16-token units
    # a hit lands a unit short of the boundary: window U).
    "gdn_fat-aligned-tp1": (
        _pack(family="gdn_fat", frontier=False, prefix_match_unit=None),
        _ALIGNED,
    ),
    "gdn_mtp-tp1": (_pack(family="gdn_mtp"), _MID),
    "dense-tp1": (
        _pack(family="dense", max_num_batched_tokens=_SMALL_BUDGET),
        _ATTENTION,
    ),
    "swa-tp1": (
        _pack(family="swa", max_num_batched_tokens=_SMALL_BUDGET),
        _ATTENTION,
    ),
}


def _mamba_managers(probe: Probe) -> list[MambaManager]:
    managers = probe.scheduler.kv_cache_manager.coordinator.single_type_managers
    return [m for m in managers if isinstance(m, MambaManager)]


def _resume_alone(llm, probe: Probe, prompt: list[int], params) -> tuple[str, Result]:
    """Run `prompt` alone, step by step: while its restore is pending, no
    Mamba manager holds anything for it (the room's count changes nothing).
    Returns the scheduler's id for it, and its result."""
    engine = llm.llm_engine
    external_id = f"estimate-{next(_ids)}"
    engine.add_request(external_id, TokensPrompt(prompt_token_ids=prompt), params)
    request_id = probe.internal_id(external_id)
    managers = _mamba_managers(probe)
    tier = probe.tier
    out = None
    pending_steps = 0
    deadline = time.monotonic() + _RESUME_TIMEOUT_S
    while out is None:
        assert time.monotonic() < deadline, (
            f"{request_id} not done after {_RESUME_TIMEOUT_S} s: "
            f"restore {tier._restores.get(request_id)}, "
            f"waits {tier._last_wait.get(request_id)}"
        )
        for output in engine.step():
            if output.request_id == external_id and output.finished:
                out = output
        if request_id in tier._restores:
            pending_steps += 1
            for manager in managers:
                assert request_id not in manager.req_to_blocks, request_id
                assert request_id not in manager._checkpoints, request_id
                assert request_id not in manager._unwritten_blocks, request_id
    # Its restore was pending, and checked, at least one step.
    assert pending_steps > 0, request_id
    completion = out.outputs[0]
    logprobs = None
    if completion.logprobs is not None:
        logprobs = [
            {token: lp.logprob for token, lp in position.items()}
            for position in completion.logprobs
        ]
    return request_id, Result(
        prompt, list(completion.token_ids), out.num_cached_tokens or 0, logprobs
    )


def _rungs(prompt_tokens: int, point: int, block_size: int, first: int) -> list[int]:
    """The checkpoint ladder's rungs past `point` (`_rungs`)."""
    end_block = prompt_tokens // block_size
    rungs, distance = [], first
    while end_block - distance > point // block_size:
        rungs.append((end_block - distance) * block_size)
        distance *= 2
    return rungs


@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_estimate_exact(engines, case, record_property):
    spec, shapes = _CASES[case]
    engine = engines.get(spec)
    probe, llm = engine.probe, engine.llm
    block_size = probe.block_size
    budget = spec.max_num_batched_tokens
    mid = any(shape.mid_block for shape in shapes)
    blocks = _MID_BLOCKS if mid else _BOUNDARY_BLOCKS
    sessions = _sessions(probe, spec, len(shapes), blocks, seed=700)
    first = run(llm, [s.prompt for s in sessions], greedy(max_tokens=1))
    for session, result in zip(sessions, first):
        session.record(result)
    record_property("fillers", _push_to_ram(llm, probe, spec, sessions))
    params = greedy(max_tokens=4, logprobs=5)
    unit = probe.resume_unit
    resumed: list[tuple[list[int], Result]] = []
    for shape, session in zip(shapes, sessions):
        floor = len(session.prompt) - len(session.prompt) % unit - unit
        prompt = session.next_tool_turn(shape.turn_blocks)
        before = probe.counters()
        request_id, result = _resume_alone(llm, probe, prompt, params)
        changed = delta(before, probe.counters())
        tier, point = probe.tier.recent_resumes[request_id]
        record = dict(shape=shape.name, point=point, tokens=len(prompt), **changed)
        for name, value in record.items():
            record_property(f"{shape.name}.{name}", value)
        save_result(f"room estimate {case}", shape.name, record)

        # The shape happened.
        assert tier == "ram", record
        assert changed.get("restore.completed", 0) == 1, record
        assert (point % block_size != 0) == shape.mid_block, record
        new = len(prompt) - point
        if shape.budget_cut:
            assert new > budget, record
        if shape.rung_cut:
            chunk_end = point + min(new, budget)
            assert any(
                point < r < chunk_end
                for r in _rungs(len(prompt), point, block_size, spec.ram_ladder_blocks)
            ), record
        # The room was exact.
        assert changed.get("restore.extra_exact_admissions", 0) == 1, record
        assert changed.get("restore.extra_short", 0) == 0, record
        assert changed.get("restore.extra_unused_blocks", 0) == 0, record
        assert changed.get("restore.extra_zero_chunk", 0) == 0, record
        assert changed.get("restore.admit_steps_0", 0) == 1, record
        assert changed.get("restore.admitted_hit_short", 0) == 0, record
        assert changed.get("spill.unspilled_evicted_blocks", 0) == 0, record
        assert result.num_cached_tokens >= floor, (record, floor)
        session.record(result)
        resumed.append((prompt, result))
    if spec.model.weights == "random":
        prompts = [p for p, _ in resumed]
        for cold, (_, cached) in zip(run(llm, prompts, params, cold=True), resumed):
            assert_outputs_close(cold, cached)


@pytest.mark.parametrize("tp", [1], ids=lambda tp: f"tp{tp}")
def test_estimate_never_short_with_urgent(engines, tp, record_property):
    """Restored agent turns (priority 10) beside an urgent prompt several
    chunks long: while it prefills, the low-priority cap cuts the agents'
    chunks, so their room may be too large, never short."""
    spec = _pack(
        family="gdn_fat",
        chat_slots=1,
        low_priority_prefill_tokens=256,
        tp=tp,
    )
    engine = engines.get(spec)
    probe, llm = engine.probe, engine.llm
    sessions = _sessions(probe, spec, 2, 8, seed=800)
    for session, result in zip(sessions, run(llm, [s.prompt for s in sessions])):
        session.record(result)
    record_property("fillers", _push_to_ram(llm, probe, spec, sessions))
    urgent = Session.start(
        seed=899,
        vocab_size=spec.model.vocab_size,
        block_size=probe.block_size,
        num_blocks=8,
        unit=probe.hash_block_size,
    )
    before = probe.counters()
    prompts = [s.next_tool_turn(1) for s in sessions] + [urgent.prompt]
    results = run(llm, prompts, greedy(max_tokens=4), priorities=[10, 10, 0])
    changed = delta(before, probe.counters())
    for name, value in sorted(changed.items()):
        record_property(name, value)
    save_result("room estimate urgent", f"tp{tp}", changed)
    assert changed.get("restore.completed", 0) >= 1, changed
    assert changed.get("restore.extra_short", 0) == 0, changed
    assert changed.get("spill.unspilled_evicted_blocks", 0) == 0, changed
    assert results[0].num_cached_tokens > 0, changed
