# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RamGreedyDualSizeFrequency on a RAM block pool with hand-built sessions: a
page group and three state groups, 1600-token blocks, a fake GPU tier and a
fake clock. Each test checks one property against the pool's own LRU order,
which the `lru` arm keeps."""

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.hybrid_checkpoint.cost import PrefillCostModel
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    make_block_hash_with_group_id,
)
from vllm.v1.simple_kv_offload.ram_policy import RamGreedyDualSizeFrequency

pytestmark = pytest.mark.cpu_test

BLOCK = 1600
PAGE = 0
STATES = (1, 2, 3)
HALF_LIFE = 600.0


def unit_key(chain: str, position: int) -> BlockHash:
    return BlockHash(f"{chain}@{position}".encode())


class Session:
    """A chain of pages up to `end`, states at `states_at` (multiples of the
    block), the first `shared` tokens under `prefix`'s keys."""

    def __init__(
        self,
        name: str,
        end: int,
        states_at: tuple[int, ...],
        *,
        prefix: str | None = None,
        shared: int = 0,
    ) -> None:
        self.name = name
        self.key = BlockHash(f"{name}:end".encode())
        self.end = end
        self.units = [
            (p, unit_key(prefix if prefix and p <= shared else name, p))
            for p in range(BLOCK, end + 1, BLOCK)
        ]
        self.states_at = states_at
        self.shared = shared

    def blocks(
        self, *, own_only: bool = True
    ) -> list[tuple[BlockHashWithGroupId, int]]:
        keys = []
        for position, unit in self.units:
            if own_only and position <= self.shared:
                continue
            groups = (PAGE, *STATES) if position in self.states_at else (PAGE,)
            keys += [(make_block_hash_with_group_id(unit, g), position) for g in groups]
        return keys


class Ram:
    def __init__(self, num_blocks: int, policy: str = "gdsf") -> None:
        # One more for the null block.
        self.pool = BlockPool(
            num_blocks + 1, enable_caching=True, hash_block_size=BLOCK
        )
        self.gpu: set[BlockHashWithGroupId] = set()
        self.time = 0.0
        self.policy = RamGreedyDualSizeFrequency(
            self.pool,
            cost=PrefillCostModel(1e-3, 1e-3),
            page_group_id=PAGE,
            state_group_ids=list(STATES),
            block_size=BLOCK,
            half_life_s=HALF_LIFE,
            on_gpu=self.gpu.__contains__,
            now=lambda: self.time,
        )
        if policy == "gdsf":
            self.pool.eviction_policy = self.policy
        self.gdsf = policy == "gdsf"

    def store(self, keys: list[tuple[BlockHashWithGroupId, int]]) -> None:
        """Copy blocks in under `keys`, evicting what the policy picks."""
        blocks = self.pool.get_new_blocks(len(keys))
        for block, (key, num_tokens) in zip(blocks, keys):
            self.pool._insert_block_hash(key, block, num_tokens=num_tokens)
        self.pool.free_blocks(blocks)
        self.policy.stored([k for k, _ in keys], [b.block_id for b in blocks])

    def add(self, session: Session, parent: Session | None = None) -> None:
        """`session` finished: RAM holds its own blocks."""
        self.store(session.blocks())
        self.policy.add_session(
            session.key,
            session.end,
            session.units,
            None if parent is None else parent.key,
        )

    def evict(self, num_blocks: int) -> None:
        """Make room for `num_blocks` blocks of something else, once the GPU
        dropped the blocks RAM copied (copies are not evicted before)."""
        self.policy._fresh.clear()
        self.policy._fresher.clear()
        filler = Session(f"filler{self.time}-{num_blocks}", num_blocks * BLOCK, ())
        self.store(filler.blocks())
        self.time += 1.0

    def held(self, session: Session) -> list[int]:
        """Positions of `session`'s own pages RAM still holds."""
        cached = self.pool.cached_block_hash_to_block
        return [
            p
            for p, unit in session.units
            if p > session.shared
            and cached.get_one_block(make_block_hash_with_group_id(unit, PAGE))
        ]

    def whole(self, session: Session) -> bool:
        return len(self.held(session)) == len(
            [p for p, _ in session.units if p > session.shared]
        )


def two_sessions(ram: Ram) -> tuple[Session, Session]:
    """A then B, equal sizes (ten pages, states at the 8th and 10th)."""
    a = Session("a", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK))
    b = Session("b", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK))
    ram.add(a)
    ram.add(b)
    return a, b


@pytest.mark.parametrize("policy", ["gdsf", "lru"])
def test_a_session_resumed_often_outlives_one_resumed_once(policy: str):
    """A is older but resumed three times, B once: when RAM must drop one,
    GDSF drops B. LRU drops A, the older."""
    ram = Ram(32, policy)
    a, b = two_sessions(ram)
    for _ in range(3):
        ram.policy.resumed(a.key)
    ram.policy.resumed(b.key)
    ram.evict(16)
    kept, lost = (a, b) if ram.gdsf else (b, a)
    assert ram.whole(kept) and not ram.held(lost)


def test_at_equal_frequency_the_cheaper_tokens_go_first():
    """Equal sizes and frequency: A's ten blocks are the first 16k tokens of
    a context, B's ten the 16k after an 80k prefix on the GPU. Deep tokens
    cost more to recompute (attention grows with context): A goes."""
    ram = Ram(32)
    prefix = Session("prefix", 50 * BLOCK, (50 * BLOCK,))
    for key, _ in prefix.blocks():
        ram.gpu.add(key)
    owner = Session("c", 50 * BLOCK, (50 * BLOCK,), prefix="prefix", shared=50 * BLOCK)
    ram.policy.add_session(owner.key, owner.end, owner.units, None)
    a = Session("a", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK))
    b = Session(
        "b", 60 * BLOCK, (58 * BLOCK, 60 * BLOCK), prefix="prefix", shared=50 * BLOCK
    )
    ram.add(b)
    ram.add(a)
    ram.evict(16)
    assert ram.whole(b) and not ram.held(a)


def test_frequency_decays():
    """A was resumed four times three half-lives ago (a count of 0.5 now), B
    once just now: A goes."""
    ram = Ram(32)
    a, b = two_sessions(ram)
    for _ in range(4):
        ram.policy.resumed(a.key)
    ram.time += 3 * HALF_LIFE
    ram.policy.resumed(b.key)
    assert ram.policy._freq(ram.policy.sessions[a.key], ram.time) == pytest.approx(0.5)
    ram.evict(16)
    assert ram.whole(b) and not ram.held(a)


def test_the_next_turn_inherits_the_count_and_retires_the_last():
    """A's next turn (A2, which extends A's chain) starts with A's count, 4,
    and takes over A's blocks as A retires. An idle session of equal size
    then goes before A2."""
    ram = Ram(35)
    a = Session("a", 8 * BLOCK, (6 * BLOCK, 8 * BLOCK))
    ram.add(a)
    for _ in range(4):
        ram.policy.resumed(a.key)
    a2 = Session(
        "a2", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK), prefix="a", shared=8 * BLOCK
    )
    ram.add(a2, parent=a)
    assert a.key not in ram.policy.sessions
    assert ram.policy.sessions[a2.key].freq == pytest.approx(4.0)
    idle = Session("idle", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK))
    ram.add(idle)
    # A's pages are A2's prefix now; A2 owns them.
    assert ram.policy.owners[unit_key("a", BLOCK)] == {a2.key}
    ram.evict(16)
    assert not ram.held(idle)
    assert ram.whole(a) and ram.whole(a2)


def test_victims_are_a_sessions_tail():
    """A session loses its deepest blocks first and never a middle page."""
    ram = Ram(16)
    s = Session("s", 10 * BLOCK, (4 * BLOCK, 10 * BLOCK))
    ram.add(s)
    ram.evict(5)
    held = ram.held(s)
    assert held == [p for p, _ in s.units][: len(held)]
    assert max(held) < 10 * BLOCK


def test_ram_blocks_under_a_gpu_only_page_are_not_a_tail():
    """The GPU alone holds S's sixth page; RAM holds the pages around it and
    the end's state. Only the blocks past the sixth page may go: losing one
    under it would cut the chain under the GPU's page."""
    ram = Ram(12)
    s = Session("s", 10 * BLOCK, (10 * BLOCK,))
    keys = s.blocks()
    gpu_page = make_block_hash_with_group_id(unit_key("s", 6 * BLOCK), PAGE)
    ram.gpu.add(gpu_page)
    ram.store([(k, n) for k, n in keys if k != gpu_page])
    ram.policy.add_session(s.key, s.end, s.units, None)
    tail = 4 + len(STATES)
    ram.evict(tail)
    assert ram.held(s) == [p for p in range(BLOCK, 6 * BLOCK, BLOCK)]


def test_blocks_past_a_hole_go_before_any_value():
    """S is resumed often, but its fifth page is in neither tier: its pages
    past the hole resume nothing and go before an idle session's tail."""
    ram = Ram(31)
    s = Session("s", 10 * BLOCK, (4 * BLOCK, 10 * BLOCK))
    hole = make_block_hash_with_group_id(unit_key("s", 5 * BLOCK), PAGE)
    ram.store([(k, n) for k, n in s.blocks() if k != hole])
    ram.policy.add_session(s.key, s.end, s.units, None)
    for _ in range(5):
        ram.policy.resumed(s.key)
    idle = Session("idle", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK))
    ram.add(idle)
    ram.evict(5 + len(STATES))
    assert ram.whole(idle)
    assert ram.held(s) == [BLOCK, 2 * BLOCK, 3 * BLOCK, 4 * BLOCK]
    assert ram.policy.stats.dead == 5 + len(STATES)


def test_a_session_with_nothing_of_its_own_retires():
    """B's blocks are all A's (a turn that added nothing): B retires once its
    exclusive tail is gone, and A's blocks become A's alone again."""
    ram = Ram(40)
    a = Session("a", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK))
    ram.add(a)
    b = Session("b", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK), prefix="a", shared=10 * BLOCK)
    ram.policy.add_session(b.key, b.end, b.units, None)
    assert b.key not in ram.policy.sessions
    assert ram.policy.owners[unit_key("a", BLOCK)] == {a.key}
    assert ram.policy.stats.retired == 1
