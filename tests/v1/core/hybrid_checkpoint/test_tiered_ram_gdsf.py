# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TieredCPUOffloadConnector's RAM GDSF policy on the spec-decode device
harness: a session's frequency follows it from the GPU into RAM and back,
RAM drops only a session's tail and exact-end states whole, and with more
sessions than both tiers hold, fewer tokens are recomputed than under RAM
LRU."""

import random

import pytest

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.simple_kv_offload.ram_policy import RamGreedyDualSizeFrequency

from .spec_device import Engine
from .test_tiered_cpu_tier import BLOCK, assert_nothing_lost, make_engine, manager
from .utils import prompt

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def gdsf(engine: Engine) -> RamGreedyDualSizeFrequency:
    policy = manager(engine).ram_policy
    assert policy is not None
    return policy


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_frequency_follows_a_session_into_ram_and_back(async_scheduling: bool):
    """A session resumed three times on the GPU keeps that count once other
    sessions push it into RAM; resuming it from RAM counts a fourth, which
    its next turn inherits."""
    engine = make_engine(async_scheduling, BLOCK, ram_eviction="gdsf")
    policy = gdsf(engine)
    tokens = prompt(1, 1200)
    for turn in range(4):
        engine.add(f"s{turn}", tokens, 17)
        engine.run(f"s{turn}")
        tokens = list(engine.requests[f"s{turn}"].all_token_ids) + prompt(10 + turn, 40)
    key = next(reversed(policy.sessions))
    assert policy.sessions[key].freq == pytest.approx(3.0, rel=0.05)
    for s in range(6):
        engine.add(f"f{s}", prompt(50 + s, 1300), 17)
        engine.run(f"f{s}")
    assert policy._freq(policy.sessions[key], policy.now()) == pytest.approx(
        3.0, rel=0.05
    )
    restores = manager(engine).restore_stats.completed
    engine.add("s4", tokens, 17)
    engine.run("s4")
    assert manager(engine).restore_stats.completed > restores
    assert key not in policy.sessions
    assert policy.sessions[next(reversed(policy.sessions))].freq == pytest.approx(
        4.0, rel=0.05
    )
    assert_nothing_lost(engine)


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_a_turn_that_parts_from_the_last_prompt_is_its_sessions_resume(
    async_scheduling: bool,
):
    """A reasoning model's next turn drops the reasoning, and its history
    renders the last prompt's generation prefix differently: it parts from
    that prompt a few tokens before its end. It still counts as that
    session's resume, recomputing from the prompt's tail state only, and
    the session's next turn inherits the count."""
    engine = make_engine(async_scheduling, BLOCK // 4, ram_eviction="gdsf")
    tokens = prompt(3, 1000)
    engine.add("t0", tokens, 40)
    engine.run("t0")
    engine.add("t1", tokens[:-3] + prompt(4, 60), 17)
    engine.run("t1")
    stats = manager(engine).resume_stats
    assert stats.counts == {"gpu": 1}, stats
    assert stats.tokens["gpu"] < BLOCK // 4, stats
    policy = gdsf(engine)
    assert policy.sessions[next(reversed(policy.sessions))].freq == pytest.approx(
        1.0, rel=0.05
    )


def hot_and_cold(engine: Engine, seed: int, *, rounds: int = 6) -> int:
    """Nine sessions, three of them resumed every round and the rest every
    third, long and short contexts. Returns the tokens their resumes
    recompute."""
    rng = random.Random(seed)
    lengths = {s: 1300 if s % 3 == 0 else 700 for s in range(9)}
    hot = {0, 1, 2}
    tokens = {s: prompt(seed * 100 + s, lengths[s]) for s in range(9)}
    recomputed = 0
    for round_ in range(rounds):
        order = [s for s in range(9) if s in hot or round_ % 3 == 0]
        rng.shuffle(order)
        for s in order:
            request_id = f"q{s}r{round_}"
            engine.add(request_id, tokens[s], rng.choice((9, 33)))
            engine.run(request_id)
            if round_:
                recomputed += len(tokens[s]) - engine.admissions[request_id][0]
            tokens[s] = list(engine.requests[request_id].all_token_ids) + prompt(
                seed * 1000 + 10 * s + round_, rng.randrange(8, 40)
            )
    return recomputed


class TailChecker:
    """Checks every RAM eviction: a live session loses no RAM block it needs
    while a deeper one stays (`tail`), and every exact-end state in the RAM
    catalog is whole after each RAM allocation (`catalog`)."""

    def __init__(self, engine: Engine) -> None:
        self.tier = manager(engine)
        self.policy = gdsf(engine)
        self.violations: list[tuple] = []
        self.evictions = 0
        take_sessions = self.policy._take_sessions

        def checked_take_sessions(take, victims: list[KVCacheBlock], num_blocks):
            before = self._needed()
            start = len(victims)
            take_sessions(take, victims, num_blocks)
            self._check_tail(before, {b.block_id for b in victims[start:]})

        self.policy._take_sessions = checked_take_sessions  # type: ignore[method-assign]
        pool = self.tier.cpu_block_pool
        get_new_blocks = pool.get_new_blocks

        def checked_get_new_blocks(num_blocks: int) -> list[KVCacheBlock]:
            blocks = get_new_blocks(num_blocks)
            self._check_catalog()
            return blocks

        pool.get_new_blocks = checked_get_new_blocks  # type: ignore[method-assign]

    def _present(self, unit: BlockHash, group_id: int) -> bool:
        key = make_block_hash_with_group_id(unit, group_id)
        cpu = self.tier.cpu_block_pool.cached_block_hash_to_block
        return cpu.get_one_block(key) is not None or self.policy.on_gpu(key)

    def _reach(self, units: list[tuple[int, BlockHash]]) -> int:
        page, *states = self.policy.group_ids
        reach = 0
        for position, unit in sorted(units):
            if not self._present(unit, page):
                if not position % BLOCK:
                    break
                continue
            if all(self._present(unit, g) for g in states):
                reach = position
        return reach

    def _needed(self) -> list[tuple[BlockHash, list[tuple[int, int]]]]:
        """Per live session, its exclusive RAM blocks up to where it resumes:
        (position, block id), a block under several of its keys at the
        deepest (a prompt tail's page that became a full block's)."""
        policy = self.policy
        cpu = self.tier.cpu_block_pool.cached_block_hash_to_block
        needed = []
        for session in policy.sessions.values():
            reach = self._reach(session.units)
            deepest: dict[int, int] = {}
            for position, unit in session.units:
                if position > reach or policy.owners.get(unit) != {session.key}:
                    continue
                for g in policy.group_ids:
                    key = make_block_hash_with_group_id(unit, g)
                    for block in cpu.get_all_blocks(key):
                        deepest[block.block_id] = max(
                            position, deepest.get(block.block_id, 0)
                        )
            needed.append((session.key, [(p, b) for b, p in deepest.items()]))
        return needed

    def _check_tail(self, before, victims: set[int]) -> None:
        if victims:
            self.evictions += 1
        for key, blocks in before:
            taken = [p for p, block_id in blocks if block_id in victims]
            if not taken:
                continue
            lowest = min(taken)
            for position, block_id in blocks:
                if position > lowest and block_id not in victims:
                    self.violations.append(("tail", key, lowest, position))

    def _check_catalog(self) -> None:
        cpu = self.tier.cpu_block_pool.cached_block_hash_to_block
        for unit in self.tier.ram_frontiers.entries.values():
            for group_id in unit.groups:
                key = make_block_hash_with_group_id(unit.key, group_id)
                if cpu.get_one_block(key) is None:
                    self.violations.append(("catalog", unit.key, group_id))


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("seed", [1, 2])
def test_ram_drops_only_tails_and_whole_exact_end_states(
    seed: int, async_scheduling: bool
):
    """More sessions than both tiers hold: every RAM eviction takes a
    session's deepest blocks first, and the RAM catalog's exact-end states
    stay whole."""
    engine = make_engine(
        async_scheduling, BLOCK // 4, num_cpu_blocks=110, ram_eviction="gdsf"
    )
    checker = TailChecker(engine)
    hot_and_cold(engine, seed, rounds=4)
    assert_nothing_lost(engine)
    assert checker.evictions > 0
    assert gdsf(engine).stats.sessions > 0, gdsf(engine).stats
    assert not checker.violations, checker.violations[:10]


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_gdsf_recomputes_fewer_tokens_than_lru(async_scheduling: bool):
    """The same oversubscribed sessions under RAM LRU and RAM GDSF: GDSF
    keeps the hot sessions and the long contexts, and recomputes at least
    15% fewer tokens."""
    totals = {}
    for policy in ("lru", "gdsf"):
        totals[policy] = sum(
            hot_and_cold(
                make_engine(
                    async_scheduling,
                    BLOCK // 4,
                    num_cpu_blocks=160,
                    ram_eviction=policy,
                ),
                seed,
            )
            for seed in (1, 2)
        )
    assert totals["gdsf"] <= 0.85 * totals["lru"], totals
