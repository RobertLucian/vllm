# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An exact-end state reaches RAM whole or not at all.

A finished turn's exact end is one key over several blocks: the attention
page it ends in and a recurrent-state block per Mamba group. The GPU drops
the state whole as soon as one of those blocks is reused, so with
`spill="inline"` the step that reuses one has to copy them all, or none: part
of a state in RAM is blocks nothing can resume from, and a state listed with
a block missing would resume a session from another request's bytes.

Two sessions each end a turn inside a block, and unrelated prompts then run
one engine step at a time. After every step, each state with a block gone
from the GPU is either in RAM in every group, with the bytes read on the GPU
and listed in `ram_frontiers`, or in RAM in none and not listed. A session
whose state RAM holds then resumes within a unit of its end, and every next
turn matches a cold run.
"""

import pytest

from .checks import assert_outputs_close
from .engine import EngineSpec
from .probes import Probe
from .workloads import Session, Stepwise, fillers, greedy, run


def _spec(family: str, async_scheduling: bool = True) -> EngineSpec:
    return EngineSpec(
        family=family,
        connector="tiered",
        spill="inline",
        async_scheduling=async_scheduling,
        # Room for every block the fillers push out.
        ram_blocks=256,
    )


SPECS = [_spec("gdn"), _spec("gdn_mtp"), _spec("gdn", async_scheduling=False)]

# Half a block past a boundary: a turn of one token ends inside a block, its
# state from a prefill (test_room_estimate.py).
_SESSION_BLOCKS = (4.5, 6.5)
_MAX_FILLERS = 48
_FILL_TOKENS = 4


class _Units:
    """After each step, where each exact-end state with a block gone from the
    GPU is."""

    def __init__(self, probe: Probe, states: dict):
        self.probe = probe
        # Key -> group id -> its GPU block, and that block's bytes.
        self.states = states
        self.original = {
            key: dict(zip(blocks, probe.digests("gpu", list(blocks.values()))))
            for key, blocks in states.items()
        }
        # Key -> "whole" or "none", from the step a block of it left the GPU.
        self.outcome: dict = {}
        self.errors: list[str] = []

    def in_ram(self, key) -> dict[int, int]:
        """Group id -> the RAM block under `key`, for the groups RAM holds."""
        ids = {g: self.probe.key_block_id(key, g, "ram") for g in self.states[key]}
        return {g: block_id for g, block_id in ids.items() if block_id is not None}

    def on_step(self) -> None:
        probe = self.probe
        if probe.tier.has_pending_stores():
            # A block taken outside a scheduling pass: its copy runs with
            # the next step, and is read then.
            return
        listed = probe.ram_exact_ends()
        for key, blocks in self.states.items():
            if key not in self.outcome and all(
                probe.key_block_id(key, g, "gpu") is not None for g in blocks
            ):
                continue
            name = bytes(key)[:6].hex()
            held = self.in_ram(key)
            if not held:
                if key in listed:
                    self.errors.append(f"{name}: listed, but RAM holds no block")
                self.outcome[key] = "none"
            elif len(held) < len(blocks):
                self.errors.append(
                    f"{name}: RAM holds groups {sorted(held)} of {sorted(blocks)}"
                )
            else:
                if key not in listed:
                    self.errors.append(f"{name}: whole in RAM, but not listed")
                if self.outcome.get(key) != "whole":
                    self.check_bytes(key)
                self.outcome[key] = "whole"

    def check_bytes(self, key) -> None:
        held = self.in_ram(key)
        copies = dict(zip(held, self.probe.digests("ram", list(held.values()))))
        wrong = [g for g in held if copies[g] != self.original[key][g]]
        if wrong:
            self.errors.append(
                f"{bytes(key)[:6].hex()}: groups {wrong} differ in RAM from the "
                "bytes their GPU blocks held"
            )


@pytest.mark.parametrize("engine", SPECS, indirect=True, ids=lambda s: s.id)
def test_exact_end_lands_whole_or_not_at_all(engine, record_property):
    probe, llm, spec = engine.probe, engine.llm, engine.spec
    bs = probe.block_size
    vocab = spec.model.vocab_size
    sessions, keys = [], []
    for i, num_blocks in enumerate(_SESSION_BLOCKS):
        session = Session.start(
            seed=40 + i,
            vocab_size=vocab,
            block_size=bs,
            num_blocks=num_blocks,
            unit=probe.hash_block_size,
        )
        known = set(probe.exact_ends())
        session.record(run(llm, [session.prompt])[0])
        new = [key for key in probe.exact_ends() if key not in known]
        assert new, f"session {i} ({len(session.prompt)} tokens) left no exact end"
        sessions.append(session)
        keys.append(new)
    # A step runs before the bytes are read: a state published from a draft
    # column is folded into place by the next one.
    run(llm, fillers(40, 1, 0.25, vocab, bs))
    on_gpu = probe.exact_ends()
    assert all(key in on_gpu for new in keys for key in new), on_gpu.keys()
    watch = _Units(probe, {key: on_gpu[key] for new in keys for key in new})
    record_property("groups", [sorted(blocks) for blocks in watch.states.values()])

    prompts = fillers(41, _MAX_FILLERS, 2, vocab, bs)
    used = 0
    while len(watch.outcome) < len(watch.states):
        if used == _MAX_FILLERS:
            pytest.fail(
                f"{used} fillers left exact-end states on the GPU: "
                f"{watch.outcome}; {probe.counters()}"
            )
        stepper = Stepwise(llm)
        stepper.submit(prompts[used], greedy(max_tokens=_FILL_TOKENS))
        stepper.finish(watch.on_step)
        used += 1
        assert not watch.errors, (watch.errors[:8], probe.counters())
    record_property("fillers", used)
    record_property("outcomes", sorted(watch.outcome.values()))
    # Nothing overwrote the copies since.
    probe.wait_for_copies()
    for key, outcome in watch.outcome.items():
        if outcome == "whole":
            watch.check_bytes(key)
    assert not watch.errors, (watch.errors[:8], probe.counters())

    params = greedy(max_tokens=4, logprobs=5)
    for session, new in zip(sessions, keys):
        end = len(session.prompt)
        listed = probe.ram_exact_ends()
        in_ram = any(key in listed for key in new)
        nxt = session.next_tool_turn(0.5)
        second = run(llm, [nxt], params)[0]
        record_property(f"resumed_at_{end}", second.num_cached_tokens)
        if in_ram:
            assert end - probe.resume_unit <= second.num_cached_tokens <= end, (
                probe.residency(nxt),
                probe.counters(),
            )
        if spec.model.weights == "random":
            cold = run(llm, [nxt], params, cold=True)[0]
            assert_outputs_close(cold, second)
