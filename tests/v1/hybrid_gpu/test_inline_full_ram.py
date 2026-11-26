# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline copies with a RAM pool too small for the sessions.

The lazy walk answers a full RAM pool by holding allocations
(test_guard_release.py). With `spill="inline"` nothing is held: a block the
GPU reuses is copied if RAM can take it, by evicting what it values less, and
leaves without a copy otherwise. test_guard_release's agents run on its pools
(RAM half the GPU's size), and on a RAM pool a quarter of that:
- they keep making progress, and no allocation is held (a restore may
  still wait for room in RAM for the copies its blocks cost);
- what RAM lists is what the GPU held: the sessions' keys are followed from
  step to step, a key's bytes read while only the GPU holds it, and any RAM
  block listed under it later must carry them, when first seen there and
  when read again later: a RAM block handed to a new copy while a key still
  lists it would differ;
- `unspilled_evicted_blocks` counts only blocks that left with no copy: no
  more than the GPU blocks seen to lose a key RAM does not hold. It need
  not be 0 on either pool: while a restore pins its sources RAM can lack a
  free block for a step, where the walk would have held the allocation.
"""

import time
from collections import deque
from dataclasses import dataclass

import pytest

from .checks import delta
from .engine import EngineSpec
from .probes import Probe
from .test_guard_release import _AGENTS, _MAX_SECONDS, _SECONDS
from .workloads import Session, drive

_RAM = 32
_SMALL_RAM = 8


def _spec(
    ram_blocks: int,
    family: str = "gdn_fat",
    production: bool = False,
    tp: int = 1,
    gpu_blocks: int = 64,
) -> EngineSpec:
    return EngineSpec(
        family=family,
        connector="tiered",
        spill="inline",
        # Production's RAM policy and scheduling.
        ram_eviction="gdsf" if production else "lru",
        async_scheduling=production,
        gpu_blocks=gpu_blocks,
        ram_blocks=ram_blocks,
        tp=tp,
    )


SPECS = (
    [_spec(_RAM, tp=tp) for tp in (1, 2)]
    + [_spec(_RAM, production=True, tp=tp) for tp in (1, 2)]
    # A dense session is its pages alone: on 64 GPU blocks the six agents'
    # live sessions all fit, and only ended ones (not followed) reach RAM
    # (GPU window C1b compared no copy).
    + [_spec(_RAM, family="dense", gpu_blocks=32)]
    + [_spec(_SMALL_RAM, production=production) for production in (False, True)]
)

# Blocks whose bytes are read per step, on either tier: reading is the
# watch's cost (a gdn_fat block is about 17 MiB).
_BUDGET = 4


@dataclass
class _Copy:
    """A key's bytes, read while only GPU block `gpu_block` held it."""

    gpu_block: int
    digest: tuple[str, ...]
    # The RAM block last found to carry them.
    ram_block: int | None = None


class _Watch:
    """Follows the agents' keys from the GPU to RAM between steps, and counts
    the GPU blocks that lose a key RAM does not hold."""

    def __init__(self, probe: Probe, sessions: list[Session]):
        self.probe = probe
        # `drive` puts a new session in an agent's place when one ends.
        self.sessions = sessions
        self.stamps: list[tuple[int, int] | None] = [None] * len(sessions)
        self.keys: list[list[tuple]] = [[] for _ in sessions]
        self.copies: dict[tuple, _Copy] = {}
        # Keys whose RAM copy was found right, to be read again.
        self.again: deque[tuple] = deque()
        self.checked = 0
        self.rechecked = 0
        # GPU blocks read, and steps a copy still to send put the checks off.
        self.read = 0
        self.unsent_steps = 0
        self.gpu_keys = probe.gpu_keys()
        self.uncopied = 0
        self.errors: list[str] = []
        self.steps = 0
        self.watch_s = 0.0

    def on_step(self, started: float) -> None:
        begun = time.monotonic()
        self.steps += 1
        self._count_uncopied()
        self._follow_copies()
        self.watch_s += time.monotonic() - begun

    def _count_uncopied(self) -> None:
        probe = self.probe
        now = probe.gpu_keys()
        for block_id, keys in self.gpu_keys.items():
            lost = keys - now.get(block_id, frozenset())
            self.uncopied += any(not probe.in_ram(key) for key in lost)
        self.gpu_keys = now

    def _session_keys(self, session: Session) -> list[tuple]:
        """(block hash, group id) of the session's attention pages, and of
        the recurrent states either tier holds for it."""
        probe = self.probe
        hashes = probe.block_hashes(session.sequence)
        keys = []
        for g in probe.full_attention_group_ids:
            per = probe.group_block_sizes[g] // probe.hash_block_size
            keys += [(h, g) for h in hashes[per - 1 :: per]]
        for g in probe.mamba_group_ids:
            keys += [
                (h, g)
                for h in hashes
                if probe.key_block_id(h, g, "gpu") is not None
                or probe.key_block_id(h, g, "ram") is not None
            ]
        return keys

    def _follow_copies(self) -> None:
        probe = self.probe
        read: list[tuple[tuple, int]] = []
        check: list[tuple[tuple, int]] = []
        for slot, session in enumerate(self.sessions):
            if not session.sequence:
                continue
            stamp = (session.seed, session.turns)
            if self.stamps[slot] != stamp:
                self.stamps[slot] = stamp
                self.keys[slot] = self._session_keys(session)
            for key in self.keys[slot]:
                gpu = probe.key_block_id(*key, "gpu")
                ram = probe.key_block_id(*key, "ram")
                copy = self.copies.get(key)
                if ram is None:
                    if gpu is None or copy is None or copy.gpu_block != gpu:
                        # In neither tier, or in a GPU block not read yet.
                        self.copies.pop(key, None)
                        if gpu is not None:
                            read.append((key, gpu))
                    else:
                        copy.ram_block = None
                elif copy is not None:
                    if gpu is not None and gpu != copy.gpu_block:
                        # Restored, or computed again, which need not give
                        # the same bytes: a later copy may be of either.
                        del self.copies[key]
                    elif ram != copy.ram_block:
                        check.append((key, ram))
        # First the copies not compared yet, then one found right earlier,
        # then keys to read on the GPU. No copy while one is still to go to
        # the workers (a block taken outside a scheduling pass): it runs with
        # the next step.
        unsent = probe.tier.has_pending_stores()
        self.unsent_steps += unsent
        check = [] if unsent else check[:_BUDGET]
        while self.again and len(check) < _BUDGET and not unsent:
            key = self.again.popleft()
            copy = self.copies.get(key)
            ram = probe.key_block_id(*key, "ram")
            if copy is not None and ram is not None and ram == copy.ram_block:
                check.append((key, ram))
                self.rechecked += 1
                break
        read = read[: _BUDGET - len(check)]
        if check:
            digests = probe.digests("ram", [ram for _, ram in check])
            for (key, ram), digest in zip(check, digests):
                copy = self.copies[key]
                if digest != copy.digest:
                    self.errors.append(
                        f"step {self.steps}: RAM block {ram} lists group "
                        f"{key[1]} key {bytes(key[0])[:6].hex()} without the "
                        f"bytes GPU block {copy.gpu_block} held"
                    )
                    del self.copies[key]
                    continue
                self.checked += copy.ram_block != ram
                copy.ram_block = ram
                self.again.append(key)
        if read:
            digests = probe.digests("gpu", [gpu for _, gpu in read])
            for (key, gpu), digest in zip(read, digests):
                self.copies[key] = _Copy(gpu, digest)
            self.read += len(read)


@pytest.mark.parametrize("engine", SPECS, indirect=True, ids=lambda s: s.id)
def test_full_ram_holds_nothing_and_lists_true_copies(engine, record_property):
    probe, llm, spec = engine.probe, engine.llm, engine.spec
    sessions = [
        Session.start(
            seed=300 + i,
            vocab_size=spec.model.vocab_size,
            block_size=probe.block_size,
            num_blocks=3,
            unit=probe.hash_block_size,
        )
        for i in range(_AGENTS)
    ]
    watch = _Watch(probe, sessions)
    before = probe.counters()
    started = time.monotonic()
    turns = drive(
        llm,
        sessions,
        seconds=_SECONDS,
        turn_blocks=1,
        resume_unit=probe.resume_unit,
        max_blocks=12,
        restart_blocks=3,
        on_step=watch.on_step,
    )
    elapsed = time.monotonic() - started
    changed = delta(before, probe.counters())
    record_property("turns", len(turns))
    record_property("seconds", round(elapsed, 1))
    record_property("watch_seconds", round(watch.watch_s, 1))
    record_property("copies_checked", watch.checked)
    record_property("copies_rechecked", watch.rechecked)
    record_property("gpu_blocks_read", watch.read)
    record_property("uncopied_blocks_seen", watch.uncopied)
    for name, value in sorted(changed.items()):
        record_property(name, value)

    assert len(turns) >= 2 * _AGENTS, f"agents stalled: {len(turns)} turns"
    assert elapsed <= _MAX_SECONDS, f"livelock: {elapsed:.0f} s, {changed}"
    held = {
        name: changed[name]
        for name in (
            "spill.held_allocations",
            "spill.hold_episodes",
            "spill.guard_releases",
        )
        if changed.get(name, 0)
    }
    assert not held, changed
    assert not watch.errors, (watch.errors[:8], changed)
    unspilled = changed.get("spill.unspilled_evicted_blocks", 0)
    assert unspilled <= watch.uncopied, (unspilled, watch.uncopied, changed)
    if spec.ram_blocks >= _RAM:
        # The run did what it checks: RAM copies were compared.
        watched = (
            f"{watch.steps} steps, {watch.read} GPU blocks read, "
            f"{watch.unsent_steps} steps with a copy still to send"
        )
        assert watch.checked > 0, (watched, changed)
