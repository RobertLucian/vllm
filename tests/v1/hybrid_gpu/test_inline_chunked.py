# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restores swap in chunks with inline copies, a bounded number unacknowledged.

A restore the GPU lacks room for loads in chunks: each chunk takes GPU
blocks, whose contents go to RAM, and its loaded RAM copies are the next
chunk's room (test_chunked_swap.py). With `spill="inline"` the blocks a chunk
takes are copied in that same step and nothing waits for the copies, so the
swap is paced by its loads alone: at most two chunks may have loads the
workers have not acknowledged.

test_chunked_swap's agents run whole (`restore_chunk_blocks=0`) and in
chunks of 8, both inline. The chunked run must swap in chunks without giving
up the rest of any restore, never have more than two chunks unacknowledged,
resume every turn where its session's last prompt ended and let no block
leave the GPU uncopied. The session blocks RAM evicted in each run are
recorded side by side.
"""

import pytest

from .test_chunked_swap import _CHUNK, _agents, _spec

_MAX_UNACKED_CHUNKS = 2


@pytest.mark.parametrize("tp", [1, 2], ids=lambda tp: f"tp{tp}")
def test_inline_restores_swap_in_chunks(engines, tp, record_property):
    whole = _agents(
        engines, _spec(tp, 0).replace(spill="inline"), record_property, "whole"
    )
    spec = _spec(tp, _CHUNK).replace(spill="inline")
    chunked = _agents(engines, spec, record_property, "chunked")
    # The chunked run's engine: its maxima since it started.
    peaks = engines.get(spec).probe.peaks()
    for name, value in sorted(peaks.items()):
        record_property(f"chunked.{name}", value)
    for name in ("ram.sessions", "ram.session_blocks", "ram.sessions_lost"):
        record_property(
            f"whole_vs_chunked.{name}", (whole.get(name, 0), chunked.get(name, 0))
        )

    assert whole.get("restore.chunked", 0) == 0, whole
    assert chunked.get("restore.chunked", 0) >= 1, chunked
    assert chunked.get("restore.chunks", 0) > chunked["restore.chunked"], chunked
    assert chunked.get("restore.chunked_aborted", 0) == 0, chunked
    assert "chunks_unacked_max" in peaks, peaks
    assert peaks["chunks_unacked_max"] <= _MAX_UNACKED_CHUNKS, (peaks, chunked)
    for arm, changed in (("whole", whole), ("chunked", chunked)):
        assert changed.get("spill.inline_spilled_blocks", 0) > 0, (arm, changed)
