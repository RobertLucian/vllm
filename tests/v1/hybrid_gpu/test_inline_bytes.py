# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An inline copy carries the bytes its block held before the step that
reuses it.

With `spill="inline"` a cached GPU block goes to RAM in the very step that
hands it to another request, and that step also zeroes the block as a new
attention page and writes it. The copy has to come first on the device. A
session ends a turn, and the bytes of its attention pages and of the
recurrent states at its end are read on the GPU on every rank. Unrelated
prompts then run one engine step at a time. After each step, every one of
those keys that left the GPU in it must be in RAM with the bytes read before,
while a page's GPU block no longer has them where a live request's attention
group took it: the block was reused and zeroed in that step, not later, and
the copy still got the session's bytes. The session's next turn then resumes
at its end with the outputs of a full prefill.

A copy that ran after the step zeroed its new pages puts zeros in RAM, and
one taken of a block its new owner already writes puts that request's bytes
there: both fail the comparison here. The fillers generate several tokens, so
the blocks they take stay in use while later steps copy others.
"""

import pytest

from .checks import assert_outputs_close
from .engine import EngineSpec
from .probes import Probe
from .test_copy_path import _bytes, _session_blocks
from .workloads import Session, Stepwise, fillers, greedy, run


def _spec(family: str, tp: int = 1, async_scheduling: bool = True) -> EngineSpec:
    return EngineSpec(
        family=family,
        connector="tiered",
        spill="inline",
        async_scheduling=async_scheduling,
        # RAM keeps every block the fillers push out: none of the session's
        # copies goes before its next turn.
        ram_blocks=256,
        tp=tp,
    )


SPECS = [_spec(family) for family in ("gdn", "gdn_fat", "gdn_mtp", "dense")] + [
    _spec("gdn_fat", tp=2),
    _spec("gdn_fat", async_scheduling=False),
]

# Filler prompts run two at a time, at most this many times.
_MAX_ROUNDS = 16
_FILL_TOKENS = 8


class _Watch:
    """After each step, the session's keys that left the GPU in it: where
    their bytes are now."""

    def __init__(self, probe: Probe, tokens: list[int], state_units: dict[int, int]):
        self.probe = probe
        self.tokens = tokens
        self.state_units = state_units
        self.gpu = _session_blocks(probe, tokens, "gpu", state_units)
        self.original = _bytes(probe, "gpu", self.gpu)
        # Key -> the RAM block that took it (None: no copy).
        self.left: dict = {}
        # Keys whose GPU block was a live request's zeroed attention page
        # after the step they left in.
        self.zeroed = 0
        # Keys that left with their block neither held nor changed after the
        # step: nothing shows it was reused there.
        self.unclaimed = 0
        # Steps not read: a copy was still to go to the workers.
        self.deferred = 0
        self.errors: list[str] = []

    def on_step(self) -> None:
        probe = self.probe
        if probe.tier.has_pending_stores():
            # A block taken outside a scheduling pass: its copy runs with
            # the next step, before that step writes, and is read then.
            self.deferred += 1
            return
        on_gpu = _session_blocks(probe, self.tokens, "gpu", self.state_units)
        gone = [k for k, v in on_gpu.items() if v is None and k not in self.left]
        if not gone:
            return
        in_ram = _session_blocks(probe, self.tokens, "ram", self.state_units)
        holders = probe.holders()
        copies = _bytes(
            probe, "ram", {k: in_ram[k] for k in gone if in_ram[k] is not None}
        )
        now = _bytes(probe, "gpu", {k: self.gpu[k] for k in gone})
        for key in gone:
            block_id = self.gpu[key]
            self.left[key] = in_ram[key]
            if key not in copies:
                self.errors.append(f"{key} left GPU block {block_id} with no RAM copy")
            elif copies[key] != self.original[key]:
                self.errors.append(
                    f"{key}: RAM block {in_ram[key]} does not hold the bytes GPU "
                    f"block {block_id} held before the step that reused it"
                )
            rewritten = now[key] != self.original[key]
            taken_as_page = holders.get(block_id) in probe.full_attention_group_ids
            if taken_as_page and key[1] != "state":
                # A new attention page is zeroed in the step that takes it: a
                # page's bytes change there (a state's block may hold zeros
                # where a page goes, and change only once written).
                if rewritten:
                    self.zeroed += 1
                else:
                    self.errors.append(
                        f"{key}: GPU block {block_id} is a live request's new "
                        "page and still holds the session's bytes"
                    )
            elif not rewritten and not probe.referenced(block_id):
                self.unclaimed += 1

    @property
    def pages_left(self) -> bool:
        return all(k in self.left for k in self.gpu if k[1] != "state")


@pytest.mark.parametrize("engine", SPECS, indirect=True, ids=lambda s: s.id)
def test_inline_copy_precedes_the_reuse(engine, record_property):
    probe, llm, spec = engine.probe, engine.llm, engine.spec
    bs = probe.block_size
    vocab = spec.model.vocab_size
    session = Session.start(
        seed=17,
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
    watch = _Watch(probe, session.prompt, state_units)
    assert None not in watch.gpu.values(), probe.residency(session.prompt)
    record_property("state_units", state_units)

    prompts = fillers(17, 2 * _MAX_ROUNDS, 4, vocab, bs)
    rounds = 0
    while not watch.pages_left:
        if rounds == _MAX_ROUNDS:
            pytest.fail(
                f"{2 * rounds} fillers left the session on the GPU: "
                f"{probe.residency(session.prompt)}; {probe.counters()}"
            )
        stepper = Stepwise(llm)
        for prompt in prompts[2 * rounds : 2 * rounds + 2]:
            stepper.submit(prompt, greedy(max_tokens=_FILL_TOKENS))
        stepper.finish(watch.on_step)
        rounds += 1
        assert not watch.errors, (watch.errors[:8], probe.counters())
    record_property("fillers", 2 * rounds)
    record_property("keys_left", len(watch.left))
    record_property("zeroed_in_their_step", watch.zeroed)
    record_property("left_without_reuse", watch.unclaimed)
    record_property("steps_deferred", watch.deferred)
    # The run did what it checks: a page was copied in the step that zeroed
    # its block.
    assert watch.zeroed > 0, (watch.left, probe.counters())

    # Nothing overwrote the copies since.
    probe.wait_for_copies()
    in_ram = _session_blocks(probe, session.prompt, "ram", state_units)
    stored = {k: in_ram[k] for k in watch.left}
    assert None not in stored.values(), probe.residency(session.prompt)
    assert _bytes(probe, "ram", stored) == {k: watch.original[k] for k in stored}, (
        "stored bytes differ"
    )

    nxt = session.next_tool_turn(0.5)
    second = run(llm, [nxt], greedy(max_tokens=4, logprobs=5))[0]
    record_property("resumed_at", second.num_cached_tokens)
    end = len(first.prompt)
    assert end - probe.resume_unit <= second.num_cached_tokens <= end, probe.residency(
        nxt
    )
    if spec.model.weights == "random":
        cold = run(llm, [nxt], greedy(max_tokens=4, logprobs=5), cold=True)[0]
        assert_outputs_close(cold, second)
