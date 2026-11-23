# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restores the GPU lacks room for swap in chunks, and agents lose nothing.

The GPU makes room for a restore by copying its own blocks to RAM first.
Restored whole, the session's RAM blocks stay pinned as the load's sources
until it lands, so RAM evicts as many blocks of other sessions for those
copies (GPU window 5, 2026-11-23: 80 restores of 58 blocks on average). In
chunks, each chunk's loaded RAM copies are the next chunk's room. Agents run
out of phase on a GPU pool far smaller than their sessions (as in
test_spill_safety), whole (`restore_chunk_blocks=0`) and in chunks of 8: the
chunked run must swap in chunks without giving up, every turn must resume
where its session's last prompt ended, and no block may leave the GPU
uncopied. What RAM evicted in each is recorded side by side."""

import pytest

from .checks import delta
from .engine import EngineSpec
from .workloads import Session, drive

_AGENTS = 8
_SECONDS = 40.0
_CHUNK = 8


def _spec(tp: int, chunk: int) -> EngineSpec:
    return EngineSpec(
        family="gdn_fat",
        connector="tiered",
        ram_eviction="gdsf",
        async_scheduling=True,
        gpu_blocks=96,
        ram_blocks=384,
        restore_chunk_blocks=chunk,
        tp=tp,
    )


def _agents(engines, spec: EngineSpec, record_property, arm: str) -> dict:
    engine = engines.get(spec)
    probe, llm = engine.probe, engine.llm
    sessions = [
        Session.start(
            seed=500 + i,
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
    record_property(f"{arm}.turns", len(turns))
    record_property(f"{arm}.short_resumes", len(short))
    for name, value in sorted(changed.items()):
        record_property(f"{arm}.{name}", value)
    assert not short, (
        arm,
        [(t.session, t.floor, t.result.num_cached_tokens) for t in short[:8]],
    )
    assert changed.get("spill.unspilled_evicted_blocks", 0) == 0, (arm, changed)
    return changed


@pytest.mark.parametrize("tp", [1, 2], ids=lambda tp: f"tp{tp}")
def test_restores_swap_in_chunks_and_lose_nothing(engines, tp, record_property):
    whole = _agents(engines, _spec(tp, 0), record_property, "whole")
    chunked = _agents(engines, _spec(tp, _CHUNK), record_property, "chunked")
    assert whole.get("restore.chunked", 0) == 0, whole
    assert chunked.get("restore.chunked", 0) >= 1, chunked
    assert chunked.get("restore.chunks", 0) > chunked["restore.chunked"], chunked
    assert chunked.get("restore.chunked_aborted", 0) == 0, chunked
