# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The same agents on upstream's eager mirror and on the tiered connector.

Equal pools: the eager mirror's RAM holds what the GPU holds, so it gets the
tiered arm's GPU and RAM blocks together as RAM. Both arms record turns,
short resumes and tokens recomputed; the tiered arm must lose nothing (no
uncopied eviction, every turn at its session's last prompt end). The eager
arm is measured, not judged: it drops a finishing hybrid request's states
by design.
"""

import pytest

from .checks import delta
from .engine import EngineSpec
from .workloads import Session, drive

_GPU, _RAM = 96, 384
_AGENTS = 8
_SECONDS = 30.0


def _arms(family: str, tp: int) -> dict[str, EngineSpec]:
    return {
        "eager": EngineSpec(
            family=family,
            connector="eager",
            gpu_blocks=_GPU,
            ram_blocks=_GPU + _RAM,
            tp=tp,
        ),
        "tiered": EngineSpec(
            family=family, connector="tiered", gpu_blocks=_GPU, ram_blocks=_RAM, tp=tp
        ),
    }


@pytest.mark.parametrize(
    ("family", "tp"),
    [("gdn_fat", 1), ("gdn_fat", 2), ("dense", 1)],
    ids=lambda v: f"tp{v}" if isinstance(v, int) else v,
)
def test_same_agents_both_tiers(engines, family, tp, record_property):
    summary = {}
    for arm, spec in _arms(family, tp).items():
        engine = engines.get(spec)
        probe = engine.probe
        sessions = [
            Session.start(
                seed=200 + i,
                vocab_size=spec.model.vocab_size,
                block_size=probe.block_size,
                num_blocks=4,
                unit=probe.hash_block_size,
            )
            for i in range(_AGENTS)
        ]
        before = probe.counters()
        turns = drive(
            engine.llm,
            sessions,
            seconds=_SECONDS,
            turn_blocks=1,
            resume_unit=probe.resume_unit,
            user_every=5,
            max_blocks=24,
        )
        short = [t for t in turns if t.short_by > 0]
        summary[arm] = {
            "turns": len(turns),
            "short_resumes": len(short),
            "full_rereads": sum(
                t.result.num_cached_tokens == 0 and t.floor > 0 for t in turns
            ),
            "recomputed_tokens": sum(t.short_by for t in short),
            "counters": delta(before, probe.counters()),
        }
        for key, value in summary[arm].items():
            record_property(f"{arm}.{key}", value)
    print(f"eager vs tiered ({family}, tp{tp}): {summary}")

    tiered = summary["tiered"]
    assert tiered["counters"].get("spill.unspilled_evicted_blocks", 0) == 0, tiered
    assert tiered["short_resumes"] == 0, tiered
