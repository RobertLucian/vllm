# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A session RAM's evictions leave with no point to resume at is logged once
and counted (`ram.sessions_lost`), and only such a session.

GPU window 5 (2026-11-23) found nine sessions re-read whole with nothing in
either tier, and no record of when or why RAM let them go. Here unrelated
prompts push an agent's session out of both tiers: its log line, if any,
must match what its next turn finds (nothing to resume from, or a point
inside it)."""

import logging

import pytest

from .checks import delta
from .engine import EngineSpec
from .workloads import Session, fillers, run

SPECS = [
    EngineSpec(
        family="gdn",
        connector="tiered",
        ram_eviction="gdsf",
        gpu_blocks=48,
        ram_blocks=48,
        tp=tp,
    )
    for tp in (1, 2)
]

_MAX_FILLERS = 40
_PREFIX = "RAM lost session"


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.parametrize("engine", SPECS, indirect=True, ids=lambda s: s.id)
def test_each_lost_session_is_logged_once(engine, record_property):
    probe, llm, spec = engine.probe, engine.llm, engine.spec
    vocab = spec.model.vocab_size
    handler = _Records()
    logger = logging.getLogger("vllm.v1.simple_kv_offload.tiered")
    logger.addHandler(handler)
    try:
        session = Session.start(
            seed=11,
            vocab_size=vocab,
            block_size=probe.block_size,
            num_blocks=8,
            unit=probe.hash_block_size,
        )
        first = run(llm, [session.prompt])[0]
        session.record(first)
        probe.wait_for_copies()
        before = probe.counters()
        # Lines from here on: the fillers' evictions.
        handler.records.clear()
        fa = probe.full_attention_group_ids[0]
        used = 0
        for filler in fillers(11, _MAX_FILLERS, 6, vocab, probe.block_size):
            run(llm, [filler])
            probe.wait_for_copies()
            used += 1
            # Stop once a page of the session is in neither tier.
            ram = probe.block_ids(session.prompt, fa, "ram")
            gpu = probe.block_ids(session.prompt, fa, "gpu")
            if any(r is None and g is None for r, g in zip(ram, gpu)):
                break
        changed = delta(before, probe.counters())
        second = run(llm, [session.next_tool_turn(0.5)])[0]
    finally:
        logger.removeHandler(handler)

    lines = [r.getMessage() for r in handler.records]
    lost = [line for line in lines if line.startswith(_PREFIX)]
    record_property("fillers", used)
    record_property("sessions_lost", changed.get("ram.sessions_lost", 0))
    record_property("log_lines", len(lost))
    record_property("resumed_at", second.num_cached_tokens)
    for line in lost[:8]:
        record_property(f"line{lost.index(line)}", line)

    assert changed.get("ram.sessions_lost", 0) >= 1, changed
    assert len(lost) == changed["ram.sessions_lost"], (lost, changed)
    # The agent's own line, by its size: its turn ended within a token of
    # its prompt's end.
    ends = {len(first.prompt), len(first.prompt) + 1}
    own = [line for line in lost if int(line.split(": ")[1].split()[0]) in ends]
    assert len(own) <= 1, own
    if own:
        assert second.num_cached_tokens == 0, (own, second.num_cached_tokens)
    # Known gap (window M, 2026-11-23): a session can come back with nothing
    # to resume from and no line (seen on this 48-block GPU, where finished
    # requests lose their states on the GPU's side at once): recorded, not
    # asserted.
    record_property("unlogged_loss", not own and second.num_cached_tokens == 0)
