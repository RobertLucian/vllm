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
- Value: ``stamp + (1 + f) * gain / size`` of its last segment, ``stamp``
  the clock (the last evicted value) when the session was last used,
  ``gain`` the seconds the segment saves (``PrefillCostModel``, growing with
  context): from the point before the deepest point it resumes at with
  blocks of either tier (or the deepest point it shares with other
  sessions) to that point, ``size`` its exclusive blocks in both tiers in the
  segment, and ``f`` its resumes, halved every half-life. A session's points
  thin out below its latest turns (a loop's start, then nothing): its tail
  costs a turn to recompute, the segment under its first point the whole
  prefill, so sessions give up their tails before any loses its prefix.
  Both tiers count alike: the GPU's least recently used blocks move to RAM
  in turn.
- A session's frequency is counted where it is resumed, whichever tier holds
  it, and its next turn's session inherits it, also after it retired.
- Victims: blocks no session owns, then RAM copies of blocks the GPU keeps
  (RAM would otherwise mirror every session restored to it), then the RAM
  blocks a session holds past the point it resumes at (nothing reads them),
  then the lowest-value
  session's exclusive blocks, deepest first, so a session loses its tail and
  never a middle page. A session whose exclusive blocks no longer resume
  deeper than what it shares retires. The blocks queued requests resume from
  go only after all of these: those requests are the next to read them. The
  back of the queue goes first, each request's deepest units first and
  whole, and first only what no block the GPU still holds rests on
  (`_take_queued`).
"""

import heapq
import itertools
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Collection, Iterable
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
# Hash units remembered as disowned, the most recent kept.
_MAX_DISOWNED = 1 << 18


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

    # The seconds all its exclusive blocks save.
    gain: float = 0.0
    # Its last segment: from the resume point before the deepest one to the
    # deepest. The seconds it saves and its exclusive blocks in both tiers.
    gain_last: float = 0.0
    size_last: int = 0
    # Its exclusive RAM blocks in the last segment by unit, deepest first,
    # and past the point it resumes at.
    live: list[list[KVCacheBlock]] = field(default_factory=list)
    dead: list[KVCacheBlock] = field(default_factory=list)


@dataclass
class RamEvictionStats:
    """Blocks the RAM pool evicted, by what they held."""

    orphans: int = 0
    sessions: int = 0
    session_blocks: int = 0
    dead: int = 0
    # RAM copies of blocks the GPU kept.
    duplicates: int = 0
    # Blocks queued requests resume from (`queued`): nothing else could go.
    queued: int = 0
    # Blocks with a key the pool took in its own LRU order: none of the above
    # could go (all in use).
    unranked: int = 0
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
        gpu_keeps: Callable[[], Callable[[BlockHashWithGroupId], bool]] | None = None,
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
        self.gpu_keeps = gpu_keeps
        self.sessions: OrderedDict[BlockHash, _Session] = OrderedDict()
        # Hash unit -> the sessions that own it.
        self.owners: dict[BlockHash, set[BlockHash]] = {}
        # Hash units whose last owner retired: what only superseded turns
        # resumed at (`disowned`).
        self._disowned: OrderedDict[BlockHash, None] = OrderedDict()
        # Hash units running requests' sessions will own (`hold`).
        self._held: Counter[BlockHash] = Counter()
        # The keys each queued request resumes from, the queue's head first,
        # all of them, and the sessions that own some of them alone
        # (`queued`).
        self._queued_requests: list[tuple[BlockHashWithGroupId, ...]] = []
        self._queued: set[BlockHashWithGroupId] = set()
        self._queued_sessions: set[BlockHash] = set()
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
            self._disowned.pop(unit, None)
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
                self._disowned[unit] = None
                self._disowned.move_to_end(unit)
                if len(self._disowned) > _MAX_DISOWNED:
                    self._disowned.popitem(last=False)
            else:
                for other in owners:
                    self._push(self.sessions[other])

    def hold(self, units: Iterable[BlockHash]) -> None:
        """A running request's session will own `units` (the resume points
        its prefill computes): until `unhold`, their RAM blocks are no
        orphans. A running request frees a state it moves past, which the
        GPU then moves to RAM."""
        for unit in units:
            self._held[unit] += 1

    def unhold(self, units: Iterable[BlockHash]) -> None:
        for unit in units:
            self._held[unit] -= 1
            if self._held[unit] <= 0:
                del self._held[unit]

    def queued(self, requests: list[tuple[BlockHashWithGroupId, ...]]) -> None:
        """The keys of the blocks each queued request resumes from, the
        queue's head first, each request's shallowest first (its pages from
        the first, then the point's own). They go only once nothing else may:
        their requests are the next to read them (`_take_queued`). A key
        several requests list is the one's nearest the head. A session whose
        deepest unit is among them, and whose alone, leaves the ranking until
        no request resumes from it: its segments are queued blocks, and
        ranking it would walk it at every eviction for nothing. (A user prompt
        after a tool loop resumes at the loop's start: its session's tail is
        no queued block and stays ranked.)"""
        listed: set[BlockHashWithGroupId] = set()
        own: list[tuple[BlockHashWithGroupId, ...]] = []
        for keys in requests:
            own.append(tuple(key for key in keys if key not in listed))
            listed.update(keys)
        self._queued_requests = own
        self._queued = listed
        sessions: set[BlockHash] = set()
        owners = self.owners
        for key in listed:
            unit = get_block_hash(key)
            unit_owners = owners.get(unit)
            if unit_owners is None or len(unit_owners) != 1:
                continue
            (owner,) = unit_owners
            session = self.sessions.get(owner)
            if session is not None and session.units and session.units[0][1] == unit:
                sessions.add(owner)
        for key in self._queued_sessions - sessions:
            session = self.sessions.get(key)
            if session is not None:
                self._push(session)
        self._queued_sessions = sessions

    @property
    def has_queued(self) -> bool:
        return bool(self._queued)

    def _is_queued(self, block: KVCacheBlock) -> bool:
        queued = self._queued
        return bool(queued) and (
            block.block_hash in queued
            or any(
                key in queued
                for key in self.pool.cached_block_hashes_by_block.get(
                    block.block_id, ()
                )
            )
        )

    def disowned(self, keys: Iterable[BlockHashWithGroupId]) -> bool:
        """Whether a block cached under `keys` holds only what retired
        sessions owned: the turns that superseded them resume past it (a
        superseded exact end, a prompt tail inside a tool loop) or not at all
        (a loop's turns whose reasoning a user prompt dropped). Nothing reads
        it again: the GPU drops it without a copy, and RAM evicts its copy
        first. A key no session ever owned (a running request's) is not."""
        found = False
        for grouped in keys:
            unit = get_block_hash(grouped)
            if unit in self.owners or unit in self._held or unit not in self._disowned:
                return False
            found = True
        return found

    def _freq(self, session: _Session, now: float) -> float:
        return self._decayed(session.freq, session.freq_at, now)

    def _decayed(self, freq: float, at: float, now: float) -> float:
        return freq * 0.5 ** ((now - at) / self.half_life_s)

    # ------------------------------------------------------------------
    # Value

    def _walk(self, session: _Session, gone: Collection[int] = ()) -> _Walk:
        """Walk `session`'s units from its start: the chain of pages either
        tier holds, the points with every group's blocks (where it can
        resume), the deepest (where it resumes) and the deepest it shares,
        and its exclusive blocks in its last segment and past it. RAM blocks
        the resume needs below a block only the GPU holds are not its tail:
        losing one would cut the chain under the GPU's part (`_Walk.live`
        stops above them). RAM blocks in `gone` (victims already chosen)
        count as evicted."""
        cpu = self.pool.cached_block_hash_to_block
        on_gpu = self.on_gpu
        key = session.key
        block_size = self.block_size
        reach = shared = previous = 0
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
                if gone and blocks:
                    blocks = [b for b in blocks if b.block_id not in gone]
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
            if position > reach:
                # (An exact-end state and a hash unit's can end at the same
                # token: one point.)
                previous, reach = reach, position
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
            last = previous < position <= reach
            if last:
                walk.size_last += held
            unlisted = [b for b in ram if b.block_id not in listed]
            listed.update(b.block_id for b in unlisted)
            if position > reach:
                walk.dead.extend(unlisted)
            elif last and position > cut and unlisted:
                walk.live.append(unlisted)
        if reach > shared:
            walk.gain = self.cost.seconds(shared, reach)
            walk.gain_last = self.cost.seconds(max(previous, shared), reach)
        return walk

    def _score(
        self, session: _Session, now: float, gone: Collection[int] = ()
    ) -> float:
        return self._scored(session, now, gone)[0]

    def _scored(
        self, session: _Session, now: float, gone: Collection[int] = ()
    ) -> tuple[float, _Walk]:
        """What evicting `session`'s last segment loses per block, and its
        walk: a session gives up its cheap tail (a turn back to its previous
        resume point) long before the prefix under its first resume point,
        which only the whole prefill brings back."""
        walk = self._walk(session, gone)
        if walk.dead:
            # RAM blocks nothing reads: before any value.
            return 0.0, walk
        if walk.gain <= 0.0:
            self._to_retire.append(session)
            return float("inf"), walk
        if not walk.live:
            # No tail in RAM to give.
            return float("inf"), walk
        value = (1.0 + self._freq(session, now)) * walk.gain_last
        return session.stamp + value / max(1, walk.size_last), walk

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
        blocks no session owns, then copies of blocks the GPU keeps, then
        sessions' blocks nothing reads, then the lowest-value sessions'
        exclusive blocks, deepest first, fresh copies last, and queued
        requests' blocks only after all of those, the back of the queue first
        (`_take_queued`). The pool takes any shortfall in its LRU order, live
        sessions' blocks included (`RamEvictionStats.unranked`)."""
        self._decay()
        queue = self.pool.free_block_queue
        victims: list[KVCacheBlock] = []
        seen: set[int] = set()
        fresh = self._fresh | self._fresher
        self._fresher, self._fresh = self._fresh, set()

        def may_go(
            block: KVCacheBlock, even_fresh: bool = False, even_queued: bool = False
        ) -> bool:
            return not (
                block.ref_cnt
                or block.block_id in seen
                or block.is_null
                or (block.block_id in fresh and not even_fresh)
                or (not even_queued and self._is_queued(block))
            )

        def take(
            block: KVCacheBlock, even_fresh: bool = False, even_queued: bool = False
        ) -> bool:
            """Takes `block` if it may go."""
            if not may_go(block, even_fresh, even_queued):
                return False
            seen.add(block.block_id)
            victims.append(block)
            return True

        extra = self.pool.cached_block_hashes_by_block

        def take_orphans() -> bool:
            """Blocks no session owns, in LRU order; fresh ones too if only
            retired sessions owned them (the GPU does not copy them again)."""
            for block in queue.iter_blocks_after(None):
                if block.block_hash is None or self._owned(block):
                    continue
                keys = (block.block_hash, *extra.get(block.block_id, ()))
                if take(block, self.disowned(keys)):
                    self.stats.orphans += 1
                    if len(victims) >= num_blocks:
                        return True
            return False

        def take_ranked() -> None:
            if not take_orphans():
                self._take_duplicates(take, victims, num_blocks)
                if len(victims) < num_blocks:
                    self._take_sessions(take, may_go, victims, num_blocks, seen)

        # Free blocks without a key first.
        for block in queue.iter_blocks_after(None):
            if block.block_hash is None and take(block) and len(victims) >= num_blocks:
                break
        else:
            take_ranked()
            if len(victims) < num_blocks:
                # Fresh copies too: the GPU copies such a block again, but
                # the pool's LRU order would take a live session's.
                fresh.clear()
                take_ranked()
            if len(victims) < num_blocks and self._queued:
                self._take_queued(take, victims, num_blocks)
        moved = [b for b in victims if b.prev_free_block is not None]
        for block in moved:
            queue.remove(block)
        queue.prepend_n(moved)
        if len(victims) < num_blocks:
            chosen = {b.block_id for b in victims}
            self.stats.unranked += sum(
                block.block_hash is not None and block.block_id not in chosen
                for block in itertools.islice(queue.iter_blocks_after(None), num_blocks)
            )
        self._retire_pending(self.now())

    def _take_queued(
        self,
        take: Callable[..., bool],
        victims: list[KVCacheBlock],
        num_blocks: int,
    ) -> None:
        """Queued requests' blocks, once nothing else may go: the back of the
        queue first, each request's deepest units first and whole, as
        `_take_sessions` takes a session's (a unit's other blocks lose the
        point with one of them gone). First only what no block of the GPU's
        rests on: a request stops at a unit the GPU still holds (it serves
        the resume, or it is on its way to RAM: losing its copy would make the
        GPU copy it again), and taking the units under it would cut the chain
        under the GPU's part. Then, if that is not enough, the units under
        the GPU's (its own still stay), rather than in the pool's LRU order,
        which would take the queue's head as soon as its back. A unit with a
        block in use stops its request either way. The sessions that lost
        blocks, by any of their keys, are ranked again: what they hold past
        the point they now resume at goes before any value (`_Walk.dead`)."""
        cpu = self.pool.cached_block_hash_to_block
        extra = self.pool.cached_block_hashes_by_block
        losers: set[BlockHash] = set()
        # Per request from the back of the queue, its units deepest first:
        # (whether the GPU holds one of its keys, its RAM blocks).
        order: list[list[tuple[bool, list[KVCacheBlock]]]] = []
        for keys in reversed(self._queued_requests):
            units: dict[BlockHash, list[BlockHashWithGroupId]] = {}
            for key in keys:
                units.setdefault(get_block_hash(key), []).append(key)
            order.append(
                [
                    (
                        any(map(self.on_gpu, unit)),
                        [b for key in unit for b in cpu.get_all_blocks(key)],
                    )
                    for unit in reversed(units.values())
                ]
            )

        def take_all(under_gpu: bool) -> bool:
            for units in order:
                for on_gpu, blocks in units:
                    if any(block.ref_cnt for block in blocks):
                        break
                    if on_gpu:
                        if under_gpu:
                            continue
                        break
                    for block in blocks:
                        if not take(block, True, True):
                            continue
                        self.stats.queued += 1
                        for key in (block.block_hash, *extra.get(block.block_id, ())):
                            losers.update(self.owners.get(get_block_hash(key), ()))
                    if len(victims) >= num_blocks:
                        return True
            return False

        if not take_all(False):
            take_all(True)
        # Valued as if the victims were evicted already.
        gone = {block.block_id for block in victims}
        now = self.now()
        for key in losers:
            session = self.sessions.get(key)
            if session is None:
                continue
            session.version = next(self._seq)
            score = self._score(session, now, gone)
            if score != float("inf"):
                heapq.heappush(self.heap, (score, session.version, key))

    def _take_duplicates(
        self,
        take: Callable[..., bool],
        victims: list[KVCacheBlock],
        num_blocks: int,
    ) -> None:
        """RAM copies of blocks the GPU keeps (`gpu_keeps`: a restored
        session that runs or stays cached, a pinned restore): losing one loses
        nothing now, and the GPU copies the block again only once it is about
        to drop it. Without this the RAM tier mirrors every session restored
        to the GPU and holds that much less of the others."""
        if self.gpu_keeps is None:
            return
        keeps = self.gpu_keeps()
        extra = self.pool.cached_block_hashes_by_block
        for block in self.pool.free_block_queue.iter_blocks_after(None):
            if block.block_hash is None:
                continue
            keys = (block.block_hash, *extra.get(block.block_id, ()))
            # Also a queued request's: the GPU block it copies stays.
            if all(keeps(key) for key in keys) and take(block, even_queued=True):
                self.stats.duplicates += 1
                if len(victims) >= num_blocks:
                    return

    def _owned(self, block: KVCacheBlock) -> bool:
        """Whether a session owns one of `block`'s keys, or will
        (`hold`)."""
        assert block.block_hash is not None
        return any(
            unit in self.owners or unit in self._held
            for unit in map(
                get_block_hash,
                (
                    block.block_hash,
                    *self.pool.cached_block_hashes_by_block.get(block.block_id, ()),
                ),
            )
        )

    def _take_sessions(
        self,
        take: Callable[..., bool],
        may_go: Callable[[KVCacheBlock], bool],
        victims: list[KVCacheBlock],
        num_blocks: int,
        gone: Collection[int],
    ) -> None:
        """The lowest-value sessions' segments, deepest first, valued as if
        the victims chosen so far (`gone`) were evicted already: a session
        that gave its tail competes again with its next segment. A unit goes
        whole or not at all: one of its blocks gone loses the point (a page
        can be the unit's and a shallower unit's too), so taking the others
        while one may not go yet (in use or fresh) would lose it unfreed."""
        heap = self.heap
        now = self.now()
        # A session that gives nothing (its blocks in use or fresh) is pushed
        # back only after this eviction: it would come straight back.
        visited: list[_Session] = []
        while heap and len(victims) < num_blocks:
            score, version, key = heap[0]
            session = self.sessions.get(key)
            if session is None or session.version != version:
                heapq.heappop(heap)
                continue
            if score > 0.0 and key in self._queued_sessions:
                # Its segments are queued blocks (`queued`): ranked again
                # once its request leaves the queue.
                heapq.heappop(heap)
                continue
            current, walk = self._scored(session, now, gone)
            if current > score + 1e-12:
                session.version = next(self._seq)
                heapq.heapreplace(heap, (current, session.version, key))
                continue
            heapq.heappop(heap)
            if current == float("inf"):
                visited.append(session)
                continue
            dead = bool(walk.dead)
            taken = len(victims)
            if dead:
                for block in walk.dead:
                    # Also a queued request's: it resumes below them now.
                    if take(block, False, True) and len(victims) >= num_blocks:
                        break
            else:
                for unit in walk.live:
                    # Deepest first: a unit that may not go yet keeps the
                    # units under it.
                    if not all(may_go(block) for block in unit):
                        break
                    for block in unit:
                        take(block)
                    if len(victims) >= num_blocks:
                        break
            if len(victims) == taken:
                visited.append(session)
                continue
            if dead:
                self.stats.dead += len(victims) - taken
            else:
                self.clock = max(self.clock, current)
                self.stats.sessions += 1
                self.stats.session_blocks += len(victims) - taken
            session.version = next(self._seq)
            rest = self._score(session, now, gone)
            if rest != float("inf"):
                heapq.heappush(heap, (rest, session.version, key))
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
        self._disowned.clear()
        self._held.clear()
        self._queued_requests = []
        self._queued = set()
        self._queued_sessions = set()
        self.heap.clear()
        self.clock = 0.0
        self._retired.clear()
        self._to_retire.clear()
        self._fresh.clear()
        self._fresher.clear()
