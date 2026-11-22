# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A session pushed out of the GPU comes back byte for byte.

The bytes of the session's attention pages and of the recurrent states at
its end are read on every rank three times: on the GPU after its turn, in
pinned RAM once the GPU has dropped them, and on the GPU again after the
next turn restored them. All three must match, and the next turn must
resume at the session's end with the outputs of a full prefill.

SimpleCPU's eager mirror (upstream's design) stores recurrent states only as
a live request hands them off and drops a finishing request's: on a hybrid
it may resume short of the end. Its arm checks the pages, and the states
only where RAM holds them.
"""

import pytest

from .checks import assert_outputs_close, delta
from .engine import EngineSpec
from .probes import Probe
from .workloads import Session, fillers, greedy, run

SPECS = [
    EngineSpec(family=family, connector=connector)
    for family in ("gdn", "gdn_fat", "gdn_mtp", "dense", "swa")
    for connector in ("eager", "tiered")
] + [
    EngineSpec(family=family, connector="tiered", tp=tp)
    for family in ("gdn", "dense")
    for tp in (2, 4, 8)
]

# Most filler prompts it may take to push the session out of the GPU.
_MAX_FILLERS = 16


def _session_blocks(
    probe: Probe, tokens: list[int], tier: str, state_units: dict[int, int]
) -> dict:
    """(group, block index or "state") -> block id in `tier`: every full-
    attention page of `tokens`, and each Mamba group's state at its unit in
    `state_units`."""
    ids = {}
    for g in probe.full_attention_group_ids:
        for i, block_id in enumerate(probe.block_ids(tokens, g, tier)):
            ids[(g, i)] = block_id
    for g, unit in state_units.items():
        ids[(g, "state")] = probe.cached_units(tokens, g, tier).get(unit)
    return ids


def _bytes(probe: Probe, tier: str, ids: dict) -> dict:
    return dict(zip(ids, probe.digests(tier, list(ids.values()))))


@pytest.mark.parametrize("engine", SPECS, indirect=True, ids=lambda s: s.id)
def test_restore_is_byte_exact(engine, record_property):
    probe, llm, spec = engine.probe, engine.llm, engine.spec
    bs = probe.block_size
    vocab = spec.model.vocab_size
    # Every state at the end must reach RAM and come back.
    whole = spec.connector == "tiered" or not spec.model.hybrid
    session = Session.start(
        seed=7,
        vocab_size=vocab,
        block_size=bs,
        num_blocks=8,
        unit=probe.hash_block_size,
    )
    first = run(llm, [session.prompt])[0]
    session.record(first)

    # The deepest state of each Mamba group: where the next turn resumes.
    state_units = {
        g: max(probe.cached_units(session.prompt, g, "gpu"), default=-1)
        for g in probe.mamba_group_ids
    }
    on_gpu = _session_blocks(probe, session.prompt, "gpu", state_units)
    assert None not in on_gpu.values(), probe.residency(session.prompt)
    original = _bytes(probe, "gpu", on_gpu)
    record_property("state_units", state_units)

    # Push the session out of the GPU until its next turn needs RAM: a page
    # gone from the GPU. Not necessarily the first one: lazy stores return
    # their blocks to the eviction end, so the fillers' fresh copies go
    # before the session's older ones (window D, dense: BBBBCCCC).
    fa_group = probe.full_attention_group_ids[0]
    used = 0
    for filler in fillers(7, _MAX_FILLERS, 4, vocab, bs):
        run(llm, [filler])
        probe.wait_for_copies()
        used += 1
        if None in probe.block_ids(session.prompt, fa_group, "gpu"):
            break
    else:
        pytest.fail(
            f"{_MAX_FILLERS} fillers left the session on the GPU: "
            f"{probe.residency(session.prompt)}; {probe.counters()}"
        )
    record_property("fillers", used)

    in_ram = _session_blocks(probe, session.prompt, "ram", state_units)
    missing = [k for k, v in in_ram.items() if v is None and (whole or k[1] != "state")]
    assert not missing, (
        f"lost from RAM after {used} fillers: {missing}; "
        f"{probe.residency(session.prompt)}"
    )
    stored = {k: v for k, v in in_ram.items() if v is not None}
    record_property("states_in_ram", sum(k[1] == "state" for k in stored))
    assert _bytes(probe, "ram", stored) == {k: original[k] for k in stored}, (
        "stored bytes differ"
    )

    counters = probe.counters()
    nxt = session.next_tool_turn(0.5)
    second = run(llm, [nxt], greedy(max_tokens=4, logprobs=5))[0]
    record_property("resumed_at", second.num_cached_tokens)
    if whole:
        end = len(first.prompt)
        assert end - probe.resume_unit <= second.num_cached_tokens <= end, (
            probe.residency(nxt)
        )
    if spec.connector == "tiered":
        assert delta(counters, probe.counters()).get("restore.completed", 0) >= 1

    restored = _session_blocks(probe, session.prompt, "gpu", state_units)
    back = {k: v for k, v in restored.items() if v is not None and k in stored}
    if whole:
        assert back.keys() == stored.keys(), probe.residency(session.prompt)
    assert _bytes(probe, "gpu", back) == {k: original[k] for k in back}, (
        "restored bytes differ"
    )

    if spec.model.weights == "random":
        cold = run(llm, [nxt], greedy(max_tokens=4, logprobs=5), cold=True)[0]
        assert_outputs_close(cold, second)
