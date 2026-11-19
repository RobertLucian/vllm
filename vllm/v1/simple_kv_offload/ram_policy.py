# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GreedyDual-Size-Frequency eviction for TieredCPUOffloadConnector's RAM pool.

A block evicted from RAM is gone from both tiers: resuming its session means
recomputing it. The policy keeps what is expensive to recompute and resumed
often, at the granularity of sessions (a finished request and the blocks it
left cached):

- A session owns the keys of its full-attention pages (block boundaries), of
  the states at those boundaries, of its prompt's tail and its end (the hash
  units inside their last blocks) and of its exact-end state. Keys another
  live session shares (a common system prompt) are owned by both; a block
  goes only when no live session owns its key or its owner is the one
  evicted. A conversation's next turn supersedes the session it resumed: the
  older session retires, and what only it owned goes first.
- Value: ``stamp + (1 + f) * gain / size``, ``stamp`` the clock (the last
  evicted value) when the session was last used, ``gain`` the seconds its
  exclusive blocks save (``PrefillCostModel``, growing with context): from
  the deepest point it shares with other sessions to the deepest point it
  resumes at with blocks of either tier, ``size`` its exclusive blocks in
  both tiers up to that point, and ``f`` its resumes, halved every half-life.
  Both tiers count alike: the GPU's least recently used blocks move to RAM
  in turn.
- A session's frequency is counted where it is resumed, whichever tier holds
  it, and its next turn's session inherits it, also after it retired.
- Victims: blocks no session owns, then the RAM blocks a session holds past
  the point it resumes at (nothing reads them), then the lowest-value
  session's exclusive blocks, deepest first, so a session loses its tail and
  never a middle page. A session whose exclusive blocks no longer resume
  deeper than what it shares retires.
"""

import heapq
import itertools
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.hybrid_checkpoint.cost import PrefillCostModel
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
)

# Sessions remembered, most recently finished or resumed kept; likewise the
# frequencies of retired ones.
_MAX_SESSIONS = 4096
# Decay steps per half-life; scores are rebuilt at each.
_DECAY_STEPS = 4


@dataclass(slots=True)
class _Session:
    key: BlockHash
    end: int
    # (position, key) of the hash units it owns, deepest first.
    units: list[tuple[int, BlockHash]]
    freq: float = 0.0
    freq_at: float = 0.0
    # The clock when it was last used: GreedyDual's aging.
    stamp: float = 0.0
    version: int = 0


@dataclass(slots=True)
class _Walk:
    """What a session's exclusive blocks are worth (`_walk`)."""

    gain: float = 0.0
    size: int = 0
    # Its exclusive RAM blocks up to the point it resumes at, deepest first,
    # and past it.
    live: list[KVCacheBlock] = field(default_factory=list)
    dead: list[KVCacheBlock] = field(default_factory=list)


@dataclass
class RamEvictionStats:
    """Blocks the RAM pool evicted, by what they held."""

    orphans: int = 0
    sessions: int = 0
    session_blocks: int = 0
    dead: int = 0
    retired: int = 0
    counts: dict[str, int] = field(default_factory=dict)


class RamGreedyDualSizeFrequency:
    """CachedBlockEvictionPolicy of the RAM pool (module docstring)."""

    def __init__(
        self,
        pool: BlockPool,
        *,
        cost: PrefillCostModel,
        page_group_id: int,
        state_group_ids: list[int],
        block_size: int,
        half_life_s: float,
        on_gpu: Callable[[BlockHashWithGroupId], bool],
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.pool = pool
        self.cost = cost
        self.now = now
        self.page_group_id = page_group_id
        self.state_group_ids = state_group_ids
        self.group_ids = [page_group_id, *state_group_ids]
        self.block_size = block_size
        self.half_life_s = half_life_s
        self.on_gpu = on_gpu
        self.sessions: OrderedDict[BlockHash, _Session] = OrderedDict()
        # Hash unit -> the sessions that own it.
        self.owners: dict[BlockHash, set[BlockHash]] = {}
        self.heap: list[tuple[float, int, BlockHash]] = []
        self.clock = 0.0
        self._seq = itertools.count()
        self._decay_every = half_life_s / _DECAY_STEPS
        self._decayed_at = self.now()
        self.stats = RamEvictionStats()
        # Retired sessions' (frequency, when): a later turn inherits it.
        self._retired: OrderedDict[BlockHash, tuple[float, float]] = OrderedDict()
        self._to_retire: list[_Session] = []
        # Blocks stored since the last two evictions. The GPU drops the
        # blocks they copy only once the copies exist: evicting a fresh copy
        # first would make the GPU copy the block again.
        self._fresh: set[int] = set()
        self._fresher: set[int] = set()

    # ------------------------------------------------------------------
    # Sessions

    def add_session(
        self,
        key: BlockHash,
        end: int,
        units: list[tuple[int, BlockHash]],
        parent: BlockHash | None,
    ) -> None:
        """A request finished at `end` tokens, owning `units` ((position,
        hash) pairs). `parent`: the session it resumed, which it supersedes
        and whose frequency it inherits."""
        now = self.now()
        session = self.sessions.pop(key, None)
        if session is not None:
            self._disown(session)
        freq = self._parent_freq(parent, now) if parent != key else 0.0
        if parent is not None and parent != key and parent in self.sessions:
            self._retire(self.sessions[parent], now)
        session = _Session(
            key, end, sorted(set(units), reverse=True), freq, now, stamp=self.clock
        )
        self.sessions[key] = session
        for _, unit in session.units:
            owners = self.owners.setdefault(unit, set())
            shared_with = list(owners)
            owners.add(key)
            for other in shared_with:
                # Its unit is no longer its own: its value changed.
                self._push(self.sessions[other])
        self._push(session)
        while len(self.sessions) > _MAX_SESSIONS:
            _, oldest = self.sessions.popitem(last=False)
            self._disown(oldest)
        self._retire_pending(now)

    def resumed(self, key: BlockHash) -> None:
        """A request resumes at the end of session `key`."""
        now = self.now()
        session = self.sessions.get(key)
        if session is None:
            retired = self._retired.get(key)
            if retired is not None:
                freq, at = retired
                self._retired[key] = (self._decayed(freq, at, now) + 1.0, now)
                self._retired.move_to_end(key)
            return
        session.freq = self._freq(session, now) + 1.0
        session.freq_at = now
        session.stamp = self.clock
        self.sessions.move_to_end(key)
        self._push(session)
        self._retire_pending(now)

    def _parent_freq(self, parent: BlockHash | None, now: float) -> float:
        if parent is None:
            return 0.0
        session = self.sessions.get(parent)
        if session is not None:
            return self._freq(session, now)
        retired = self._retired.pop(parent, None)
        return 0.0 if retired is None else self._decayed(*retired, now)

    def _retire(self, session: _Session, now: float) -> None:
        """`session` adds nothing to the sessions that share its blocks, or a
        later turn supersedes it: what only it owned goes first."""
        if self.sessions.pop(session.key, None) is None:
            return
        self._disown(session)
        self.stats.retired += 1
        self._retired[session.key] = (self._freq(session, now), now)
        self._retired.move_to_end(session.key)
        while len(self._retired) > _MAX_SESSIONS:
            self._retired.popitem(last=False)

    def _retire_pending(self, now: float) -> None:
        while self._to_retire:
            session = self._to_retire.pop()
            # Another's retiring may have left it blocks of its own.
            if session.key in self.sessions and self._walk(session).gain <= 0.0:
                self._retire(session, now)

    def _disown(self, session: _Session) -> None:
        for _, unit in session.units:
            owners = self.owners.get(unit)
            if owners is None:
                continue
            owners.discard(session.key)
            if not owners:
                del self.owners[unit]
            else:
                for other in owners:
                    self._push(self.sessions[other])

    def _freq(self, session: _Session, now: float) -> float:
        return self._decayed(session.freq, session.freq_at, now)

    def _decayed(self, freq: float, at: float, now: float) -> float:
        return freq * 0.5 ** ((now - at) / self.half_life_s)

    # ------------------------------------------------------------------
    # Value

    def _walk(self, session: _Session) -> _Walk:
        """Walk `session`'s units from its start: the chain of pages either
        tier holds, the deepest point with every group's blocks (where it
        resumes), the deepest such point it shares, and its exclusive blocks
        up to there and past it. RAM blocks the resume needs below a block
        only the GPU holds are not its tail: losing one would cut the chain
        under the GPU's part (`_Walk.live` stops above them)."""
        cpu = self.pool.cached_block_hash_to_block
        on_gpu = self.on_gpu
        key = session.key
        block_size = self.block_size
        reach = shared = 0
        broken = False
        # Per unit, ascending: (position, whether it is exclusive, its blocks
        # in either tier, its RAM blocks, whether the GPU alone holds its
        # page, whether the GPU alone holds one of its blocks).
        units: list[tuple[int, bool, int, list[KVCacheBlock], bool, bool]] = []
        for position, unit in reversed(session.units):
            exclusive = self.owners.get(unit) == {key}
            held = 0
            ram: list[KVCacheBlock] = []
            complete = True
            page_on_gpu_only = any_on_gpu_only = False
            for group_id in self.group_ids:
                grouped = make_block_hash_with_group_id(unit, group_id)
                blocks = cpu.get_all_blocks(grouped)
                if exclusive:
                    ram.extend(blocks)
                gpu_only = not blocks and on_gpu(grouped)
                if blocks or gpu_only:
                    held += 1
                    if gpu_only:
                        any_on_gpu_only = True
                        page_on_gpu_only |= group_id == self.page_group_id
                elif group_id == self.page_group_id and not position % block_size:
                    # The chain has a hole: nothing past it resumes.
                    broken = True
                else:
                    complete = False
            units.append(
                (position, exclusive, held, ram, page_on_gpu_only, any_on_gpu_only)
            )
            if broken or not complete:
                continue
            reach = position
            if not exclusive:
                shared = position
        # The deepest block the resume at `reach` needs that only the GPU
        # holds: a page on the way, or the state there.
        cut = -1
        for position, _, _, _, page_on_gpu_only, any_on_gpu_only in units:
            if position > reach:
                break
            if (
                position < reach and page_on_gpu_only and not position % block_size
            ) or (position == reach and any_on_gpu_only):
                cut = position
        walk = _Walk()
        # A block can hold several of the session's keys.
        listed: set[int] = set()
        for position, exclusive, held, ram, _, _ in reversed(units):
            if not exclusive:
                continue
            if position <= reach:
                walk.size += held
            into = (
                walk.dead if position > reach else walk.live if position > cut else None
            )
            for block in ram:
                if block.block_id not in listed:
                    listed.add(block.block_id)
                    if into is not None:
                        into.append(block)
        if reach > shared:
            walk.gain = self.cost.seconds(shared, reach)
        return walk

    def _score(self, session: _Session, now: float) -> float:
        walk = self._walk(session)
        if walk.dead:
            # RAM blocks nothing reads: before any value.
            return 0.0
        if walk.gain <= 0.0:
            self._to_retire.append(session)
            return float("inf")
        if not walk.live:
            # No tail in RAM to give.
            return float("inf")
        return session.stamp + (1.0 + self._freq(session, now)) * walk.gain / max(
            1, walk.size
        )

    def _push(self, session: _Session) -> None:
        session.version = next(self._seq)
        score = self._score(session, self.now())
        # A session with nothing exclusive to give is not pushed; it is again
        # when RAM stores one of its units or it loses a sharer.
        if score != float("inf"):
            heapq.heappush(self.heap, (score, session.version, session.key))

    def stored(self, keys: list[BlockHashWithGroupId], block_ids: list[int]) -> None:
        """RAM now holds blocks `block_ids` under `keys`: their owners' values
        changed. Fresh copies are not evicted before the GPU drops the blocks
        they copy (`prepare_eviction`)."""
        self._fresh.update(block_ids)
        touched = set()
        for grouped in keys:
            touched.update(self.owners.get(get_block_hash(grouped), ()))
        for key in touched:
            self._push(self.sessions[key])
        self._retire_pending(self.now())

    # ------------------------------------------------------------------
    # CachedBlockEvictionPolicy

    def on_blocks_freed(self, blocks: list[KVCacheBlock]) -> None:
        return

    def lower_tier_stored(self, keys: list[BlockHashWithGroupId]) -> None:
        return

    def peek_eviction_order(self, num_blocks: int) -> list[KVCacheBlock]:
        return []

    def prepare_eviction(self, num_blocks: int) -> None:
        """Move the next `num_blocks` victims to the free queue's front:
        blocks no session owns, then sessions' blocks nothing reads, then the
        lowest-value sessions' exclusive blocks, deepest first."""
        self._decay()
        queue = self.pool.free_block_queue
        victims: list[KVCacheBlock] = []
        seen: set[int] = set()
        fresh = self._fresh | self._fresher
        self._fresher, self._fresh = self._fresh, set()

        def take(block: KVCacheBlock) -> bool:
            """Takes `block` if it may go."""
            if (
                block.ref_cnt
                or block.block_id in seen
                or block.is_null
                or block.block_id in fresh
            ):
                return False
            seen.add(block.block_id)
            victims.append(block)
            return True

        # Free blocks without a key, then blocks no session owns, in LRU order.
        for block in queue.iter_blocks_after(None):
            if block.block_hash is None and take(block) and len(victims) >= num_blocks:
                break
        else:
            for block in queue.iter_blocks_after(None):
                if block.block_hash is None or self._owned(block):
                    continue
                if take(block):
                    self.stats.orphans += 1
                    if len(victims) >= num_blocks:
                        break
            else:
                self._take_sessions(take, victims, num_blocks)
        moved = [b for b in victims if b.prev_free_block is not None]
        for block in moved:
            queue.remove(block)
        queue.prepend_n(moved)
        self._retire_pending(self.now())

    def _owned(self, block: KVCacheBlock) -> bool:
        assert block.block_hash is not None
        if get_block_hash(block.block_hash) in self.owners:
            return True
        return any(
            get_block_hash(key) in self.owners
            for key in self.pool.cached_block_hashes_by_block.get(block.block_id, ())
        )

    def _take_sessions(
        self,
        take: Callable[[KVCacheBlock], bool],
        victims: list[KVCacheBlock],
        num_blocks: int,
    ) -> None:
        heap = self.heap
        now = self.now()
        # Each session is visited once per eviction and pushed back after:
        # its blocks stay cached until the pool pops them, and one whose
        # blocks are all in use or fresh would otherwise come straight back.
        visited: list[_Session] = []
        while heap and len(victims) < num_blocks:
            score, version, key = heap[0]
            session = self.sessions.get(key)
            if session is None or session.version != version:
                heapq.heappop(heap)
                continue
            current = self._score(session, now)
            if current > score + 1e-12:
                session.version = next(self._seq)
                heapq.heapreplace(heap, (current, session.version, key))
                continue
            heapq.heappop(heap)
            visited.append(session)
            if current == float("inf"):
                continue
            walk = self._walk(session)
            dead = bool(walk.dead)
            taken = len(victims)
            for block in walk.dead if dead else walk.live:
                if take(block):
                    if len(victims) >= num_blocks:
                        break
                elif not dead:
                    # Deepest first: a tail block that may not go yet (in
                    # use or fresh) keeps the blocks under it.
                    break
            if len(victims) == taken:
                continue
            if dead:
                self.stats.dead += len(victims) - taken
                continue
            self.clock = max(self.clock, current)
            self.stats.sessions += 1
            self.stats.session_blocks += len(victims) - taken
        for session in visited:
            if session.key in self.sessions:
                self._push(session)

    def _decay(self) -> None:
        now = self.now()
        if now - self._decayed_at < self._decay_every:
            return
        self._decayed_at = now
        # Frequencies decay continuously; scores are rebuilt at these steps.
        heap = []
        for session in self.sessions.values():
            session.version = next(self._seq)
            score = self._score(session, now)
            if score != float("inf"):
                heap.append((score, session.version, session.key))
        heapq.heapify(heap)
        self.heap = heap
        self._retire_pending(now)

    def reset(self) -> None:
        self.sessions.clear()
        self.owners.clear()
        self.heap.clear()
        self.clock = 0.0
        self._retired.clear()
        self._to_retire.clear()
        self._fresh.clear()
        self._fresher.clear()
