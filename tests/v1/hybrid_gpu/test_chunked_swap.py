# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A restore the GPU lacks room for swaps in chunks, and comes back exact.

The GPU makes room for a restore by copying its own blocks to RAM first.
Restored whole, the session's RAM blocks stay pinned as the load's sources
until it lands, so RAM evicts as many blocks of other sessions for those
copies (GPU window 5, 2026-11-23: 80 restores of 58 blocks on average).
In chunks, each chunk's loaded RAM copies are the next chunk's room. Here
a session leaves a small GPU for a full RAM and resumes, whole
(`restore_chunk_blocks=0`) and in chunks: the chunked restore must load in
several chunks without giving up, resume where the session can (its end,
or a ladder rung if RAM took its tail meanwhile), match a cold run, and
lose no block uncopied. What RAM evicts for each is recorded; at this size
it is mostly the session's own dead pages and orphans, so the comparison
of other sessions' losses comes from the production-scale trace (the
session-loss log, `ram.sessions_lost`)."""

import pytest

from .checks import assert_outputs_close, delta
from .engine import EngineSpec
from .workloads import Session, fillers, greedy, run

_CHUNK = 8
_SESSION_BLOCKS = 24
_MAX_FILLERS = 40
# Blocks other sessions lose to the swap: everything RAM evicted but copies of
# blocks the GPU keeps.
_OTHERS = (
    "ram.session_blocks",
    "ram.dead",
    "ram.orphans",
    "ram.queued",
    "ram.unranked",
)


def _spec(tp: int, chunk: int) -> EngineSpec:
    return EngineSpec(
        family="gdn_fat",
        connector="tiered",
        ram_eviction="gdsf",
        gpu_blocks=64,
        ram_blocks=48,
        restore_chunk_blocks=chunk,
        tp=tp,
    )


def _swap(engines, spec: EngineSpec, record_property, arm: str):
    """The session leaves the GPU, RAM fills, and its next turn resumes."""
    engine = engines.get(spec)
    probe, llm = engine.probe, engine.llm
    vocab = spec.model.vocab_size
    session = Session.start(
        seed=21,
        vocab_size=vocab,
        block_size=probe.block_size,
        num_blocks=_SESSION_BLOCKS,
        unit=probe.hash_block_size,
    )
    first = run(llm, [session.prompt])[0]
    session.record(first)
    fa = probe.full_attention_group_ids[0]
    for filler in fillers(21, _MAX_FILLERS, 4, vocab, probe.block_size):
        run(llm, [filler])
        probe.wait_for_copies()
        if all(b is None for b in probe.block_ids(session.prompt, fa, "gpu")):
            # The session is in RAM alone; stop once RAM is full, before it
            # has to evict the session's own blocks.
            assert probe.cpu_pool is not None
            empty = sum(
                block.block_hash is None
                for block in probe.cpu_pool.free_block_queue.iter_blocks_after(None)
            )
            if not empty:
                break
    else:
        pytest.fail(f"the session stayed on the GPU: {probe.residency(session.prompt)}")
    before = probe.counters()
    nxt = session.next_tool_turn(0.5)
    second = run(llm, [nxt], greedy(max_tokens=4, logprobs=5))[0]
    changed = delta(before, probe.counters())
    others = sum(changed.get(name, 0) for name in _OTHERS)
    record_property(f"{arm}.resumed_at", second.num_cached_tokens)
    record_property(f"{arm}.others_evicted", others)
    for name, value in sorted(changed.items()):
        record_property(f"{arm}.{name}", value)
    end = len(first.prompt)
    assert 0 < second.num_cached_tokens <= end, (arm, probe.residency(nxt))
    if spec.model.weights == "random":
        cold = run(llm, [nxt], greedy(max_tokens=4, logprobs=5), cold=True)[0]
        assert_outputs_close(cold, second)
    return changed, others


@pytest.mark.parametrize("tp", [1, 2], ids=lambda tp: f"tp{tp}")
def test_chunked_swap_evicts_no_one_for_its_own_room(engines, tp, record_property):
    whole, whole_others = _swap(engines, _spec(tp, 0), record_property, "whole")
    chunked, chunked_others = _swap(
        engines, _spec(tp, _CHUNK), record_property, "chunked"
    )
    assert whole.get("restore.chunked", 0) == 0, whole
    assert chunked.get("restore.chunked", 0) == 1, chunked
    assert chunked.get("restore.chunks", 0) >= 2, chunked
    assert chunked.get("restore.chunked_aborted", 0) == 0, chunked
    assert chunked.get("restore.completed", 0) >= 1, chunked
    record_property("others_evicted", f"whole {whole_others}, chunked {chunked_others}")
    for changed in (whole, chunked):
        assert changed.get("spill.unspilled_evicted_blocks", 0) == 0, changed
