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
    once just now: A goes. Each resumes only at its end, so each is one
    segment."""
    ram = Ram(32)
    a = Session("a", 10 * BLOCK, (10 * BLOCK,))
    b = Session("b", 10 * BLOCK, (10 * BLOCK,))
    ram.add(a)
    ram.add(b)
    for _ in range(4):
        ram.policy.resumed(a.key)
    ram.time += 3 * HALF_LIFE
    ram.policy.resumed(b.key)
    assert ram.policy._freq(ram.policy.sessions[a.key], ram.time) == pytest.approx(0.5)
    # The 6 free blocks, then A's 13.
    ram.evict(19)
    assert ram.whole(b) and not ram.held(a)


def test_sessions_give_up_their_tails_before_a_prefix():
    """A and B resume at their ends and 8 blocks in. Losing the last two
    blocks costs a session those two blocks' prefill; losing the state 8
    blocks in costs it all 8 (no point resumes below). Making room for one
    session's tail takes both tails, not one session's prefix."""
    ram = Ram(32)
    a, b = two_sessions(ram)
    ram.evict(10)
    for s in (a, b):
        assert ram.held(s) == [p for p in range(BLOCK, 8 * BLOCK + 1, BLOCK)]


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


def test_two_keys_at_a_sessions_end_are_one_point():
    """A's end holds every group's blocks under two keys (an exact-end state
    and the hash unit there): one resume point, not a segment of its own.
    A still ranks, below B (resumed three times), and loses its tail first."""
    ram = Ram(40)
    a = Session("a", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK))
    a.units.append((10 * BLOCK, BlockHash(b"a:frontier@end")))
    ram.store(a.blocks())
    ram.policy.add_session(a.key, a.end, a.units, None)
    b = Session("b", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK))
    ram.add(b)
    assert ram.policy._score(ram.policy.sessions[a.key], ram.time) < float("inf")
    for _ in range(3):
        ram.policy.resumed(b.key)
    ram.evict(10)
    assert ram.whole(b)
    held = ram.held(a)
    assert 10 * BLOCK not in held
    assert set(range(BLOCK, 8 * BLOCK + 1, BLOCK)) <= set(held)
    assert ram.policy.stats.unranked == 0


def queue(*sessions: Session) -> list[tuple[BlockHashWithGroupId, ...]]:
    """What `TieredCPUOffloadScheduler.observe_queue` passes for requests
    queued in this order (the head first), each resuming at its session's
    end: its pages from the first, then the states there."""
    requests = []
    for session in sessions:
        end = session.units[-1][1]
        keys = [make_block_hash_with_group_id(u, PAGE) for _, u in session.units]
        keys += [make_block_hash_with_group_id(end, g) for g in STATES]
        requests.append(tuple(keys))
    return requests


def store_but(ram: Ram, session: Session, gpu_only: int) -> None:
    """`session` finished with the GPU alone holding its blocks at
    `gpu_only` (the page and the states there): RAM holds the rest."""
    end = {key for key, n in session.blocks() if n == gpu_only}
    ram.gpu.update(end)
    ram.store([(key, n) for key, n in session.blocks() if key not in end])
    ram.policy.add_session(session.key, session.end, session.units, None)


def has_states(ram: Ram, session: Session, position: int) -> bool:
    """Whether RAM holds every group's state at `position` of `session`."""
    cached = ram.pool.cached_block_hash_to_block
    unit = dict(session.units)[position]
    return all(
        cached.get_one_block(make_block_hash_with_group_id(unit, g)) for g in STATES
    )


@pytest.mark.parametrize("queued", [True, False])
def test_a_queued_requests_session_goes_last(queued: bool):
    """A is older than B and resumed less, so RAM would drop A first. A
    request waiting in the queue resumes from A: RAM drops B instead, as
    that request is the next to read A's blocks."""
    ram = Ram(32)
    a, b = two_sessions(ram)
    ram.policy.resumed(b.key)
    if queued:
        ram.policy.queued(queue(a))
    ram.evict(16)
    kept, lost = (a, b) if queued else (b, a)
    assert ram.whole(kept) and not ram.held(lost)
    assert ram.policy.stats.queued == 0


def test_queued_requests_give_up_the_back_of_the_queue_first():
    """Only queued requests' blocks are left to evict, each request resuming
    at its session's end. The request queued last loses its end first (its
    last page and the states there, whole), then its 9th page, and still
    resumes at its state 8 blocks in; the queue's head keeps all of its
    own."""
    ram = Ram(32)
    a, b = two_sessions(ram)
    ram.policy.queued(queue(a, b))
    # B's end (its last page and the states there), then its 9th page.
    ram.evict(5)
    assert ram.whole(a)
    assert ram.held(b) == [p * BLOCK for p in range(1, 9)]
    assert has_states(ram, b, 8 * BLOCK)
    assert ram.policy.stats.queued == 5


def test_what_a_queued_request_can_no_longer_read_goes_first():
    """B's request waits at B's end, behind A's. With only queued blocks
    left, B gives up its end's states: it resumes 8 blocks in now, and B's
    pages past that are blocks nothing reads. Once A's request is admitted,
    those pages go before A's (which A resumes from, but no request waits
    for), although B's request still lists them."""
    ram = Ram(32)
    a, b = two_sessions(ram)
    ram.policy.queued(queue(a, b))
    ram.evict(3)
    ram.policy.queued(queue(b))
    # The 3 blocks the last eviction stored (no session's), then B's 9th
    # page and what is left of its end.
    ram.evict(5)
    assert ram.whole(a)
    assert ram.held(b) == [p * BLOCK for p in range(1, 9)]


def test_the_back_of_the_queue_goes_first_while_its_copies_are_fresh():
    """B was copied to RAM just now, A two evictions before: B's copies are
    fresh (the GPU may still hold the blocks they copy). With only queued
    blocks left, the back of the queue (B) still gives first: its copies
    being fresh is no reason to take the head's."""
    ram = Ram(33)
    a = Session("a", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK))
    b = Session("b", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK))
    ram.add(a)
    ram.store(Session("orphan", BLOCK, ()).blocks())
    ram.add(b)
    ram.policy.queued(queue(a, b))
    # Room for 5 blocks without `Ram.evict`, which forgets fresh copies: the
    # orphan, then B's end (its last page and the states there).
    ram.store(Session("filler", 5 * BLOCK, ()).blocks())
    assert ram.whole(a)
    assert ram.held(b) == [p * BLOCK for p in range(1, 10)]
    assert not has_states(ram, b, 10 * BLOCK)
    assert ram.policy.stats.queued == 4


def test_a_queued_request_keeps_the_ram_blocks_under_its_gpu_tail():
    """The GPU alone holds B's end (its last page and the states there); RAM
    holds the rest of B, and A. With only queued blocks left, B's RAM blocks
    stay although B is at the back of the queue: they are the chain under
    the GPU's part, which would resume nothing without them. A gives its
    end instead."""
    ram = Ram(28)
    a = Session("a", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK))
    b = Session("b", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK))
    ram.add(a)
    store_but(ram, b, 10 * BLOCK)
    ram.policy.queued(queue(a, b))
    ram.evict(4)
    assert ram.held(b) == [p * BLOCK for p in range(1, 10)]
    assert has_states(ram, b, 8 * BLOCK)
    assert ram.held(a) == [p * BLOCK for p in range(1, 10)]
    assert ram.policy.stats.queued == 4


def test_with_only_blocks_under_the_gpus_left_the_back_is_cut_first():
    """The GPU alone holds both sessions' ends, RAM the rest. With only
    queued blocks left, the back of the queue gives the deepest of its
    blocks under its GPU part (its pages: its requests resume at its end),
    rather than the pool's LRU order taking the head's older blocks."""
    ram = Ram(24)
    a = Session("a", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK))
    b = Session("b", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK))
    store_but(ram, a, 10 * BLOCK)
    store_but(ram, b, 10 * BLOCK)
    ram.policy.queued(queue(a, b))
    ram.evict(4)
    assert ram.held(a) == [p * BLOCK for p in range(1, 10)]
    assert has_states(ram, a, 8 * BLOCK)
    assert ram.held(b) == [p * BLOCK for p in range(1, 6)]
    assert ram.policy.stats.queued == 4
    assert ram.policy.stats.unranked == 0


def test_a_session_ranks_again_once_its_request_leaves_the_queue():
    """A's request waits, so B gives its end although B was resumed more.
    Once A's request is admitted, A ranks again: the next eviction takes
    A's end, the cheaper tail, and not B's prefix."""
    ram = Ram(32)
    a, b = two_sessions(ram)
    ram.policy.resumed(b.key)
    ram.policy.queued(queue(a))
    ram.evict(4)
    assert ram.whole(a)
    assert ram.held(b) == [p * BLOCK for p in range(1, 10)]
    ram.policy.queued([])
    # The 4 blocks the last eviction stored (no session's), B's 9th page
    # (it resumes 8 blocks in now), then A's end.
    ram.evict(9)
    assert ram.held(a) == [p * BLOCK for p in range(1, 10)]
    assert ram.held(b) == [p * BLOCK for p in range(1, 9)]


def test_the_sessions_a_queued_block_holds_a_key_of_are_ranked_again():
    """B's last page also holds a page of C (a block can carry several keys).
    Both are queued; with only queued blocks left, B's end goes, C's 3rd
    page with it. C resumes nowhere now, so nothing reads what RAM holds of
    it: it goes next, before any more of B's."""
    ram = Ram(24)
    b = Session("b", 10 * BLOCK, (8 * BLOCK, 10 * BLOCK))
    c = Session("c", 6 * BLOCK, (6 * BLOCK,))
    ram.add(b)
    shared = make_block_hash_with_group_id(unit_key("c", 3 * BLOCK), PAGE)
    ram.store([(key, n) for key, n in c.blocks() if key != shared])
    ram.policy.add_session(c.key, c.end, c.units, None)
    last_page = make_block_hash_with_group_id(unit_key("b", 10 * BLOCK), PAGE)
    block = ram.pool.cached_block_hash_to_block.get_one_block(last_page)
    ram.pool._insert_block_hash(shared, block, num_tokens=3 * BLOCK)
    ram.policy.queued(queue(c, b))
    ram.evict(4)
    assert ram.held(b) == [p * BLOCK for p in range(1, 10)]
    # The 4 blocks the last eviction stored (no session's), then what
    # nothing reads: C's 8 blocks and B's 9th page (B resumes 8 blocks in).
    ram.evict(13)
    assert not ram.held(c) and not has_states(ram, c, 6 * BLOCK)
    assert ram.held(b) == [p * BLOCK for p in range(1, 9)]
