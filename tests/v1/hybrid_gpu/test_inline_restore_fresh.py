# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A restore planned from RAM copies registered moments ago reads the right
bytes.

With `spill="inline"` RAM lists a copy the moment the scheduler plans it, and
nothing waits for an acknowledgement: a restore may pick the copy as a source
in the same scheduling pass, or the next. Its load must still run after the
copy on the device, and the RAM blocks it reads must stay as they are while
the restore takes GPU blocks, whose own contents are copied inline into RAM.

A session ends a turn and unrelated prompts run until it is next to leave the
GPU. A large prompt then takes its blocks while its next turn arrives: with
the prompt (odd seeds), or the step after the prompt took the first of them
(even seeds). The turn resumes from copies made within that run. As it
finishes, the session's blocks under its resume point must hold on the GPU
the bytes read after its first turn, and its outputs must match a cold run.
One arm's RAM is barely the session's size: the copies made for the restore
compete with its sources there (RAM may then lose the session, and the turn
re-read it: only bytes and outputs are checked).
"""

import pytest

from .checks import assert_outputs_close, delta
from .engine import Engine, EngineSpec
from .test_copy_path import _bytes, _session_blocks
from .workloads import Session, Stepwise, fillers, greedy, run

_AMPLE_RAM = 256
_SMALL_RAM = 16


def _spec(family: str, ram_blocks: int = _AMPLE_RAM, tp: int = 1) -> EngineSpec:
    return EngineSpec(
        family=family,
        connector="tiered",
        spill="inline",
        async_scheduling=True,
        ram_blocks=ram_blocks,
        tp=tp,
    )


SPECS = [
    _spec("gdn_fat"),
    _spec("gdn_fat", tp=2),
    _spec("gdn_mtp"),
    _spec("dense"),
    _spec("gdn_fat", ram_blocks=_SMALL_RAM),
]

_SEEDS = (1, 2, 3, 4)
_SESSION_BLOCKS = 8
# The prompt that takes the session's blocks: more than the session holds,
# with room left on the GPU to restore it beside the prompt.
_BIG_BLOCKS = 24
_FILL_TOKENS = 8
# Most one-block prompts it may take to bring the session to the GPU's
# eviction end.
_MAX_FILLERS = 64


def _resume_beside_eviction(engine: Engine, seed: int) -> dict:
    probe, llm, spec = engine.probe, engine.llm, engine.spec
    bs = probe.block_size
    vocab = spec.model.vocab_size
    session = Session.start(
        seed=900 + seed,
        vocab_size=vocab,
        block_size=bs,
        num_blocks=_SESSION_BLOCKS,
        unit=probe.hash_block_size,
    )
    session.record(run(llm, [session.prompt])[0])
    # The deepest state of each Mamba group: where the next turn resumes.
    state_units = {
        g: max(probe.cached_units(session.prompt, g, "gpu"), default=-1)
        for g in probe.mamba_group_ids
    }
    on_gpu = _session_blocks(probe, session.prompt, "gpu", state_units)
    assert None not in on_gpu.values(), probe.residency(session.prompt)
    original = _bytes(probe, "gpu", on_gpu)

    def off_gpu() -> set:
        now = _session_blocks(probe, session.prompt, "gpu", state_units)
        return {k for k, v in now.items() if v is None}

    # The session is next to leave the GPU once its first block has.
    prompts = fillers(900 + seed, _MAX_FILLERS, 1, vocab, bs)
    used = 0
    while not (gone_before := off_gpu()):
        if used == _MAX_FILLERS:
            pytest.fail(
                f"seed {seed}: {used} fillers left the session on the GPU: "
                f"{probe.residency(session.prompt)}"
            )
        run(llm, [prompts[used]])
        used += 1

    before = probe.counters()
    nxt = session.next_tool_turn(0.5)
    params = greedy(max_tokens=4, logprobs=5)
    stepper = Stepwise(llm)
    stepper.submit(
        fillers(950 + seed, 1, _BIG_BLOCKS, vocab, bs)[0],
        greedy(max_tokens=_FILL_TOKENS),
    )
    turn = stepper.submit(nxt, params) if seed % 2 else None
    # Keys seen off the GPU at any step, and the session's blocks on the GPU
    # as its turn finished, with their bytes.
    left = set(gone_before)
    back: dict = {}
    restored: dict = {}
    done = False

    def on_step() -> None:
        nonlocal turn, back, restored, done
        now = _session_blocks(probe, session.prompt, "gpu", state_units)
        left.update(k for k, v in now.items() if v is None)
        if turn is None and len(left) > len(gone_before):
            turn = stepper.submit(nxt, params)
        if not done and turn in stepper.results:
            done = True
            back = {k: v for k, v in now.items() if v is not None}
            restored = _bytes(probe, "gpu", back)

    stepper.finish(on_step)
    if turn is None:
        pytest.fail(
            f"seed {seed}: the large prompt took none of the session's blocks: "
            f"{probe.residency(session.prompt)}"
        )
    second = stepper.results[turn]
    changed = delta(before, probe.counters())

    # What the turn resumed from: the keys under its resume point (it
    # computed the rest again).
    def resumed_from(key) -> bool:
        group_id, index = key
        if index == "state":
            end = (state_units[group_id] + 1) * probe.hash_block_size
        else:
            end = (index + 1) * probe.group_block_sizes[group_id]
        return end <= second.num_cached_tokens

    kept = [k for k in back if resumed_from(k)]
    wrong = [k for k in kept if restored[k] != original[k]]
    assert not wrong, (
        f"seed {seed}: restored bytes differ for {wrong}; resumed at "
        f"{second.num_cached_tokens} of {len(session.prompt)}; {changed}"
    )
    if spec.model.weights == "random":
        cold = run(llm, [nxt], params, cold=True)[0]
        assert_outputs_close(cold, second, name=f"seed {seed}")
    # Blocks on the GPU as the run began, copied to RAM during it and back
    # under the turn's resume point: restored from those copies.
    fresh = [k for k in kept if k in left and k not in gone_before]
    return {
        "fillers": used,
        "resumed_at": second.num_cached_tokens,
        "of": len(session.prompt),
        "fresh_blocks": len(fresh),
        "restores": changed.get("restore.completed", 0),
        "inline_spilled": changed.get("spill.inline_spilled_blocks", 0),
    }


@pytest.mark.parametrize("engine", SPECS, indirect=True, ids=lambda s: s.id)
def test_restore_reads_fresh_copies(engine, record_property):
    runs = {}
    for seed in _SEEDS:
        # Both tiers empty again (the fixture's engine starts that way).
        engine.reset()
        runs[seed] = _resume_beside_eviction(engine, seed)
        record_property(f"seed_{seed}", runs[seed])
    if engine.spec.ram_blocks >= _AMPLE_RAM:
        # The run did what it checks: some turn resumed from copies made
        # while the large prompt was taking its session's blocks.
        assert sum(r["fresh_blocks"] for r in runs.values()) > 0, runs
