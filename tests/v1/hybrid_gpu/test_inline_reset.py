# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A reset right after inline copies were issued leaves nothing behind.

With `spill="inline"` RAM lists a copy as the scheduler plans it, before the
workers have run or acknowledged it. A reset then has to forget those copies
on the scheduler's side whatever the workers still report, and a copy issued
before it must not land in a RAM block handed out again after it.

Eight requests whose outputs outgrow the GPU pool run together: the pool
runs out, requests are preempted, and the blocks they give up are reused
while they wait, copied inline. Short prompts follow until one finishes in a
step that issued copies, and `Engine.reset()` comes straight after. Then
neither tier may hold a key of anything that ran, no exact-end state may be
listed, no store may be pending, and a new session must go to RAM and back
byte for byte (test_copy_path's test, run on this engine), with nothing of
the earlier requests turning up afterwards.
"""

import pytest

from . import test_copy_path as copy_path
from .checks import delta
from .engine import EngineSpec
from .probes import Probe
from .workloads import Stepwise, fillers, greedy


def _spec(family: str, async_scheduling: bool = True) -> EngineSpec:
    # test_copy_path's pools.
    return EngineSpec(
        family=family,
        connector="tiered",
        spill="inline",
        async_scheduling=async_scheduling,
    )


SPECS = [
    _spec("gdn"),
    _spec("gdn_mtp"),
    _spec("dense"),
    # A one-token request ends in the step that reads its prompt: the reset
    # follows the copies that step issued.
    _spec("dense", async_scheduling=False),
]

_REQUESTS = 8
_PROMPT_BLOCKS = 2
# Each output, in blocks: with the prompts, well over the 64-block GPU pool.
_GROW_BLOCKS = 8
_MAX_TAILS = 8


def _cached(probe: Probe, sequences: list[list[int]]) -> dict[int, str]:
    """The sequences either tier still caches a key of -> where."""
    out = {}
    for i, tokens in enumerate(sequences):
        residency = probe.residency(tokens)
        if set("".join(residency.marks.values())) - {"."}:
            out[i] = str(residency)
    return out


@pytest.mark.parametrize("engine", SPECS, indirect=True, ids=lambda s: s.id)
def test_reset_after_inline_copies(engine, record_property):
    probe, llm, spec = engine.probe, engine.llm, engine.spec
    bs = probe.block_size
    vocab = spec.model.vocab_size

    def spilled() -> int:
        return probe.counters().get("spill.inline_spilled_blocks", 0)

    before = probe.counters()
    preemptions: dict[str, int] = {}

    def count_preemptions() -> None:
        for request_id, request in probe.scheduler.requests.items():
            preemptions[request_id] = request.num_preemptions

    stepper = Stepwise(llm)
    for prompt in fillers(70, _REQUESTS, _PROMPT_BLOCKS, vocab, bs):
        stepper.submit(prompt, greedy(max_tokens=_GROW_BLOCKS * bs))
    stepper.finish(count_preemptions)
    used = [result.sequence for result in stepper.results.values()]
    changed = delta(before, probe.counters())
    record_property("preemptions", sum(preemptions.values()))
    for name, value in sorted(changed.items()):
        record_property(name, value)
    # The run did what the reset follows: requests were preempted, and the
    # blocks they gave up were copied inline as others took them.
    assert sum(preemptions.values()) > 0, "the workload did not preempt"
    assert changed.get("spill.inline_spilled_blocks", 0) > 0, changed

    def copies_in_last_step(prompt: list[int]) -> int:
        """Run `prompt` alone, one token: the copies its last step issued."""
        stepper = Stepwise(llm)
        stepper.submit(prompt)
        counts = [spilled()]
        stepper.finish(lambda: counts.append(spilled()))
        return counts[-1] - counts[-2]

    # A prompt whose last step issued copies, if one of these ends that way.
    in_last_step = 0
    for tail in fillers(71, _MAX_TAILS, 1, vocab, bs):
        used.append(tail)
        in_last_step = copies_in_last_step(tail)
        if in_last_step:
            break
    record_property("copies_in_last_step", in_last_step)

    engine.reset()
    assert not probe.tier.has_pending_stores()
    assert not _cached(probe, used), _cached(probe, used)
    assert not probe.exact_ends(), probe.exact_ends().keys()
    assert not probe.ram_exact_ends(), probe.ram_exact_ends()

    copy_path.test_restore_is_byte_exact(engine, record_property)
    # Nothing issued before the reset landed after it.
    probe.wait_for_copies()
    assert not probe.tier.has_pending_stores()
    assert not _cached(probe, used), _cached(probe, used)
