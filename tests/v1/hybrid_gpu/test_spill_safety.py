# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy copies never cost a session its blocks.

Eight agents run out of phase on a GPU pool far smaller than their sessions,
for four times the spill guard's 10 s patience, so allocations keep waiting
on copies while new ones arrive. No block a lookup could restore may leave
the GPU without a RAM copy, and every turn must resume where its session's
last prompt ended.

GPU window 4 (2026-11-22, Qwen3.8-27B, 55.75 MB blocks) broke both: the
guard let held allocations go eight times, 64 blocks left uncopied, and ten
resumes re-read whole 100-190k-token sessions. Fat pages (about 17 MiB) give
the copies real PCIe time here; the attention-only case checks the same
promise without recurrent states.
"""

import pytest

from .checks import delta
from .engine import EngineSpec
from .workloads import Session, drive

SPECS = [
    EngineSpec(
        family="gdn_fat", connector="tiered", gpu_blocks=96, ram_blocks=384, tp=tp
    )
    for tp in (1, 2)
] + [
    EngineSpec(family=family, connector="tiered", gpu_blocks=96, ram_blocks=384)
    for family in ("gdn", "dense")
]

_AGENTS = 8
_SECONDS = 40.0


@pytest.mark.parametrize("engine", SPECS, indirect=True, ids=lambda s: s.id)
def test_no_block_leaves_uncopied(engine, record_property):
    probe, llm, spec = engine.probe, engine.llm, engine.spec
    sessions = [
        Session.start(
            seed=100 + i,
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
    record_property("turns", len(turns))
    record_property("short_resumes", len(short))
    record_property("recomputed_tokens", sum(t.short_by for t in short))
    for name, value in sorted(changed.items()):
        record_property(name, value)

    # The guard's own path. `uncopied_evicted_blocks` also counts superseded
    # exact-end states the tier drops on purpose: recorded, not asserted.
    assert changed.get("spill.unspilled_evicted_blocks", 0) == 0, changed
    assert not short, (
        f"{len(short)} of {len(turns)} turns resumed short of their session's "
        "last prompt end (session, floor, resumed): "
        f"{[(t.session, t.floor, t.result.num_cached_tokens) for t in short[:8]]}"
    )
