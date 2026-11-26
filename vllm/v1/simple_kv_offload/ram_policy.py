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
  blocks a session holds past the point it resumes at (nothing reads them,
  unless a queued request resumes past that point),
  then the lowest-value
  session's exclusive blocks, deepest first, so a session loses its tail and
  never a middle page. A session whose exclusive blocks no longer resume
  deeper than what it shares retires. The blocks queued requests resume from
  go only after all of these: those requests are the next to read them. The
  back of the queue goes first, each request's deepest units first and
  whole, and first only what no block the GPU still holds rests on
  (`_take_queued`).
- Ladder points (`ladder_ranking`, the checkpoint ladder tiered.py keeps
  inside a session: `rungs`): their states go on their own, the pages above
  them staying. With "safe", a rung no cut could stop at goes at once: cuts
  go deepest first, and a segment worth less per block than every one above
  it goes right after them, so the point on top of it is resumed at only
  until RAM gets to that segment (`_records`); it goes only while the points
  that can be resumed at stay the same (`_safe_drops`). Cheaper victims
  (orphans, duplicates, dead pages) can put that off: in GPU window A
  (2026-11-25) a one-turn session's far rung served its next turn, and
  without it the turn re-read the session; across a busy run "safe" still
  recomputed ~12% less than keeping every rung, which held 13% of RAM. On
  production's trace (window B, same day) seven turns resumed below a rung
  it had dropped, most from the start of their session, against one that
  resumed at a rung it kept: the connector's default is "fixed". A
  rung that can be resumed at is ranked with the pages by the seconds its
  segment saves per state block, a GPU-resident
  session's too, whose tail RAM cannot take. With "expected" a rung is worth
  what losing it adds to the session's expected recompute at its next turn,
  its return raced against the clock's rise (`_expected_recompute`).
"""

import heapq
import itertools
import math
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
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
# A session with turn gaps on record keeps its resume count undecayed while
# it is not overdue (`_freq`), at most this many.
_MAX_LIVE_FREQ = 8.0
# The weight of the latest gap in a session's average (`arrived`).
_GAP_WEIGHT = 0.5
# Score floors: RAM blocks nothing reads go first, then overdue sessions'
# (`_scored`); live sessions score from 0 up.
_DEAD_SCORE = -2.0
_OVERDUE_SCORE = -1.0
# How ladder points are valued (module docstring).
LADDER_RANKINGS = ("fixed", "safe", "expected")
# The clock's rise per second is measured over this window, from samples at
# most a second apart, once it spans at least `_RATE_MIN_SPAN_S`; scores are
# rebuilt when it changes by more than `_RATE_REBUILD_RATIO`.
_RATE_WINDOW_S = 120.0
_RATE_MIN_SPAN_S = 30.0
_RATE_REBUILD_RATIO = 2.0
# Two values per block this close are a tie (`_records`).
_EPS = 1e-12
# The weight of the latest gap in the average across sessions (`_gap_avg`):
# about the last twenty.
_GAP_AVG_WEIGHT = 0.05


def _averaged(average: float, value: float, weight: float) -> float:
    """An exponentially weighted average (0: none yet)."""
    return value if not average else weight * value + (1 - weight) * average


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
    # Average seconds from a turn finishing to the next one arriving, for
    # tool-loop turns and for user turns (0: none seen), and the requests
    # that resume it now, queued or running (`arrived`, `departed`).
    gap_tool: float = 0.0
    gap_user: float = 0.0
    active: int = 0
    # Its ladder points (hash unit -> position) whose states may still go on
    # their own (`ladder_ranking`).
    rungs: dict[BlockHash, int] = field(default_factory=dict)
    # The chance its tail is cut before its next turn, as estimated when it
    # was added (None: no estimate), and the point it resumed at then: its
    # next turn checks the estimate (`ladder_p`).
    p_cut: float | None = None
    reach_at_add: int = 0


@dataclass(slots=True)
class _Point:
    """A point a session resumes at, above what it shares (`_Walk.points`)."""

    position: int
    unit: BlockHash
    # The segment under it (from the point below, or the shared point): the
    # seconds it saves and its exclusive blocks in both tiers.
    gain: float
    size: int
    # A ladder point: its state groups' blocks in either tier, and its RAM
    # ones.
    rung: bool = False
    states: int = 0
    ram_states: list[KVCacheBlock] = field(default_factory=list)


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
    # The deepest point it resumes at with blocks of either tier, and the
    # deepest it shares with other sessions.
    reach: int = 0
    shared: int = 0
    # Its points above `shared`, shallowest first: the last is `reach`.
    points: list[_Point] = field(default_factory=list)
    # Its exclusive RAM blocks: attention pages, and states.
    ram_pages: int = 0
    ram_states: int = 0


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
    # Of all the above, copies of GPU blocks about to leave (`leaving`):
    # nothing but queued requests' blocks could go otherwise. Their GPU
    # blocks leave uncopied.
    leaving: int = 0
    # Sessions these evictions left with no point of their own to resume at
    # (`settle`): their next turns re-read them whole.
    sessions_lost: int = 0
    # Blocks with a key the pool took in its own LRU order: none of the above
    # could go (all in use).
    unranked: int = 0
    retired: int = 0
    # Ladder points whose states went by value (`ladder_ranking`), their
    # blocks, and of those the copies stored since the last two evictions
    # (copied only to be dropped); ladder points no cut could stop at,
    # dropped at once (`_safe_drops`).
    rungs: int = 0
    rung_blocks: int = 0
    rung_fresh: int = 0
    rungs_safe: int = 0
    counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class SessionLoss:
    """A session RAM's evictions left with no point of its own to resume at
    (`RamGreedyDualSizeFrequency.settle`)."""

    key: BlockHash
    # Tokens its last turn ended at, and the deepest point it could resume
    # at before and after the evictions (both tiers; the latter at most the
    # point it shares with other sessions).
    end: int
    reach_before: int
    reach_after: int
    # Its blocks the evictions took, and their kinds (`prepare_eviction`).
    blocks: int
    kinds: tuple[str, ...]
    # Seconds since it was last used (its turn finished or it was resumed).
    idle_s: float


# A session's points as (position, seconds its segment saves, its blocks,
# the point's state blocks), shallowest first, the last its reach.
_Segments = list[tuple[int, float, int, int]]


def _segments(points: list[_Point]) -> _Segments:
    return [(p.position, p.gain, p.size, p.states) for p in points]


def _records(segments: _Segments) -> frozenset[int]:
    """The points below the reach a session can be resumed at: cuts take
    segments deepest first at their value per block, and once one goes, every
    segment under it worth no more per block goes too. A point is resumed at
    only if the segment under it is worth more per block than every one above
    it (stamp and frequency are the session's, the same for all)."""
    records: list[int] = []
    # Walking down: the most any segment above point k - 1 is worth.
    above = -math.inf
    for k in range(len(segments) - 1, 0, -1):
        _, gain, size, _ = segments[k]
        above = max(above, gain / max(1, size))
        position, gain_k, size_k, _ = segments[k - 1]
        if gain_k / max(1, size_k) > above + _EPS:
            records.append(position)
    return frozenset(records)


def _merged(segments: _Segments, k: int) -> _Segments:
    """`segments` without point `k`'s states: its segment joins the next."""
    position, gain, size, states = segments[k]
    upper = segments[k + 1]
    return [
        *segments[:k],
        (upper[0], gain + upper[1], size + upper[2] - states, upper[3]),
        *segments[k + 2 :],
    ]


def _safe_drops(segments: _Segments, candidates: Collection[int]) -> list[int]:
    """The ladder points among `candidates` (positions) whose states can go
    with the points the session can be resumed at staying the same
    (`_records`), deepest first, each checked with the drops before it: a
    rung under a record keeps the record worth stopping at, so of two such
    rungs one may go but not both."""
    current = list(segments)
    keep = _records(current)
    drops: list[int] = []
    for position in sorted(candidates, reverse=True):
        if position in keep:
            continue
        k = next((i for i, s in enumerate(current) if s[0] == position), None)
        if k is None or k >= len(current) - 1:
            continue
        merged = _merged(current, k)
        if _records(merged) == keep:
            current = merged
            drops.append(position)
    return drops


def _expected_recompute(
    segments: _Segments,
    level: Callable[[float, int], float],
    chance_cut: Callable[[float], float],
) -> float:
    """The seconds a session's next turn is expected to recompute: it
    resumes at point k when, at its return, the cuts reached past every
    segment above k but not the one under it, and at the shared point under
    them all past every segment. `level(seconds, blocks)` is the clock at
    which a segment goes, `chance_cut(level)` the chance the clock reaches it
    before the session returns."""
    expected = above_seconds = 0.0
    reached_above = -math.inf
    chance_above = 1.0
    for _, gain, size, _ in reversed(segments):
        # The clock at which this segment goes, at the latest with those above.
        reached = max(level(gain, size), reached_above)
        chance = chance_cut(reached)
        # Resumes at this segment's point: everything above it cut, not it.
        expected += (chance_above - chance) * above_seconds
        above_seconds += gain
        reached_above, chance_above = reached, chance
    return expected + chance_above * above_seconds


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
        abandon_gap_multiplier: float = 0.0,
        abandon_floor_s: float = 30.0,
        gpu_keeps: Callable[[], Callable[[BlockHashWithGroupId], bool]] | None = None,
        leaving: Callable[[], Iterable[KVCacheBlock]] | None = None,
        dropped: Callable[[list[KVCacheBlock]], None] | None = None,
        ladder_ranking: str = "fixed",
        may_drop_point: Callable[[BlockHash, BlockHash], bool] | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if ladder_ranking not in LADDER_RANKINGS:
            raise ValueError(
                f"ladder_ranking must be one of {LADDER_RANKINGS}, "
                f"got {ladder_ranking!r}"
            )
        self.pool = pool
        self.cost = cost
        self.now = now
        self.page_group_id = page_group_id
        self.state_group_ids = state_group_ids
        self.group_ids = [page_group_id, *state_group_ids]
        self.block_size = block_size
        self.half_life_s = half_life_s
        # A session idle for this many times its longer turn gap (at least
        # `abandon_floor_s`) is overdue: abandoned, it goes before every live
        # session (`_overdue`). 0: off, frequency halves every half-life.
        self.abandon_gap_multiplier = abandon_gap_multiplier
        self.abandon_floor_s = abandon_floor_s
        self.on_gpu = on_gpu
        self.gpu_keeps = gpu_keeps
        # Copies of GPU blocks about to leave, which the GPU waits for to drop
        # them: victims only before queued requests' blocks, and then their
        # GPU blocks may leave uncopied (`dropped`, with the victims).
        self.leaving = leaving
        self.dropped = dropped
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
        # Retired sessions' (frequency, when, tool gap, user gap): a later turn
        # inherits them.
        self._retired: OrderedDict[BlockHash, tuple[float, float, float, float]] = (
            OrderedDict()
        )
        # (when it becomes overdue, its version then, key): sessions are scored
        # again then (`_rescore_overdue`), as scores are fixed when pushed.
        self._due: list[tuple[float, int, BlockHash]] = []
        self._to_retire: list[_Session] = []
        # Blocks stored since the last two evictions. The GPU drops the
        # blocks they copy only once the copies exist: evicting a fresh copy
        # first would make the GPU copy the block again.
        self._fresh: set[int] = set()
        self._fresher: set[int] = set()
        # Sessions whose blocks evictions took since the last `settle`: the
        # session, the point it resumed at before, its blocks taken, and the
        # kinds of eviction.
        self._touched: dict[BlockHash, tuple[_Session, int, list[int], set[str]]] = {}
        # How ladder points are valued (module docstring); whether the
        # connector lets a session's point at a hash unit go on its own (one
        # only it keeps, nobody queued at it); the points chosen to go,
        # (session, unit, why), until the connector drops them
        # (`take_dropped_points`).
        self.ladder_ranking = ladder_ranking
        self.may_drop_point = may_drop_point
        self._dropped_points: list[tuple[BlockHash, BlockHash, str]] = []
        # (when, clock) at most a second apart over `_RATE_WINDOW_S`, the
        # clock's rise per second they give (None: unknown) and the one the
        # scores were last rebuilt with.
        self._clock_samples: deque[tuple[float, float]] = deque()
        self.clock_rate: float | None = None
        self._rate_built: float | None = None
        # The average turn gap across sessions: a session with none on record
        # is expected back after it (`_expected_gap`).
        self._gap_avg = 0.0
        # Sessions last scored with nothing RAM can rank, though they hold
        # blocks of their own (`unranked_blocks`).
        self._unranked: set[BlockHash] = set()
        # Sessions' estimated chance of a cut tail before their next turn, by
        # tenth, against what happened: (bucket, "cut" | "intact") -> count.
        self.ladder_p: Counter[tuple[str, str]] = Counter()

    # ------------------------------------------------------------------
    # Sessions

    def add_session(
        self,
        key: BlockHash,
        end: int,
        units: list[tuple[int, BlockHash]],
        parent: BlockHash | None,
        rungs: Mapping[BlockHash, int] | None = None,
    ) -> None:
        """A request finished at `end` tokens, owning `units` ((position,
        hash) pairs). `parent`: the session it resumed, which it supersedes
        and whose frequency it inherits. `rungs`: its ladder points (hash
        unit -> position), whose states may go on their own
        (`ladder_ranking`)."""
        now = self.now()
        session = self.sessions.pop(key, None)
        if session is not None:
            self._disown(session)
        freq = self._parent_freq(parent, now) if parent != key else 0.0
        gaps = self._parent_gaps(parent) if parent != key else (0.0, 0.0)
        if parent is not None and parent != key:
            # Inherited: a retired parent's record is used up.
            self._retired.pop(parent, None)
        if parent is not None and parent != key and parent in self.sessions:
            self._retire(self.sessions[parent], now)
        session = _Session(
            key,
            end,
            sorted(set(units), reverse=True),
            freq,
            now,
            stamp=self.clock,
            gap_tool=gaps[0],
            gap_user=gaps[1],
            rungs=dict(rungs) if rungs and self.ladder_ranking != "fixed" else {},
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
        walk = self._push(session)
        session.reach_at_add = walk.reach
        session.p_cut = self._chance_tail_cut(session, walk, now)
        self._schedule_due(session)
        while len(self.sessions) > _MAX_SESSIONS:
            _, oldest = self.sessions.popitem(last=False)
            self._disown(oldest)
        self._retire_pending(now)

    def arrived(self, key: BlockHash, at: float, extends_end: bool) -> None:
        """A request that resumes session `key` arrived at `at` (this policy's
        clock): one more gap on record, of a tool loop's turn (`extends_end`)
        or a user turn, and the session is live while the request is queued
        or runs (`departed`)."""
        session = self.sessions.get(key)
        if session is None:
            retired = self._retired.get(key)
            if retired is not None and at > retired[1]:
                freq, when, tool, user = retired
                tool, user = self._with_gap(tool, user, at - when, extends_end)
                self._retired[key] = (freq, when, tool, user)
                self._note_gap(at - when)
            return
        if at > session.freq_at:
            session.gap_tool, session.gap_user = self._with_gap(
                session.gap_tool, session.gap_user, at - session.freq_at, extends_end
            )
            self._note_gap(at - session.freq_at)
        session.active += 1
        self._push(session)

    def _note_gap(self, gap: float) -> None:
        self._gap_avg = _averaged(self._gap_avg, gap, _GAP_AVG_WEIGHT)

    def _expected_gap(self, session: _Session) -> float:
        """Seconds until `session`'s next turn arrives, as `_patience` reads
        its gaps: its longer one on record, else the average across sessions,
        else the abandonment floor."""
        return (
            max(session.gap_tool, session.gap_user)
            or self._gap_avg
            or self.abandon_floor_s
        )

    def departed(self, key: BlockHash) -> None:
        """A request `arrived` counted finished or left (`arrived`)."""
        session = self.sessions.get(key)
        if session is None or not session.active:
            return
        session.active -= 1
        if not session.active:
            self._schedule_due(session)

    @staticmethod
    def _with_gap(
        tool: float, user: float, gap: float, extends_end: bool
    ) -> tuple[float, float]:
        new = _averaged(tool if extends_end else user, gap, _GAP_WEIGHT)
        return (new, user) if extends_end else (tool, new)

    def _parent_gaps(self, parent: BlockHash | None) -> tuple[float, float]:
        if parent is None:
            return 0.0, 0.0
        session = self.sessions.get(parent)
        if session is not None:
            return session.gap_tool, session.gap_user
        retired = self._retired.get(parent)
        return (0.0, 0.0) if retired is None else (retired[2], retired[3])

    def _patience(self, session: _Session) -> float | None:
        """Seconds idle after which `session` is overdue, or None (no gap on
        record, or the rule is off)."""
        gap = max(session.gap_tool, session.gap_user)
        if not self.abandon_gap_multiplier or not gap:
            return None
        return self.abandon_gap_multiplier * max(gap, self.abandon_floor_s)

    def _overdue(self, session: _Session, now: float) -> bool:
        """Whether `session` is abandoned by its own rhythm: no request of its
        is queued or running and it has been idle longer than its patience
        (`abandon_gap_multiplier` times its longer turn gap, tool loop or
        user, at least `abandon_floor_s`: a tool call's gap is short, a
        person's long, and the same agent has both)."""
        patience = self._patience(session)
        return (
            patience is not None
            and not session.active
            and now - session.freq_at > patience
        )

    def _schedule_due(self, session: _Session) -> None:
        patience = self._patience(session)
        if patience is not None and not session.active:
            heapq.heappush(
                self._due, (session.freq_at + patience, session.version, session.key)
            )

    def _rescore_overdue(self, now: float) -> None:
        """Sessions that turned overdue since they were scored go to the
        overdue tier now."""
        while self._due and self._due[0][0] <= now:
            _, version, key = heapq.heappop(self._due)
            session = self.sessions.get(key)
            if session is not None and session.version == version:
                self._push(session)

    def resumed(self, key: BlockHash) -> None:
        """A request resumes at the end of session `key`."""
        now = self.now()
        session = self.sessions.get(key)
        if session is None:
            retired = self._retired.get(key)
            if retired is not None:
                freq, at, tool, user = retired
                self._retired[key] = (
                    self._decayed(freq, at, now) + 1.0,
                    now,
                    tool,
                    user,
                )
                self._retired.move_to_end(key)
            return
        if session.p_cut is not None:
            # Whether the estimate made when it was added came true.
            cut = self._walk(session).reach < session.reach_at_add
            bucket = f"{min(9, int(10 * session.p_cut)) / 10:.1f}"
            self.ladder_p[bucket, "cut" if cut else "intact"] += 1
            session.p_cut = None
        session.freq = self._freq(session, now) + 1.0
        session.freq_at = now
        session.stamp = self.clock
        self.sessions.move_to_end(key)
        self._push(session)
        self._schedule_due(session)
        self._retire_pending(now)

    def _parent_freq(self, parent: BlockHash | None, now: float) -> float:
        if parent is None:
            return 0.0
        session = self.sessions.get(parent)
        if session is not None:
            return self._freq(session, now)
        retired = self._retired.get(parent)
        return 0.0 if retired is None else self._decayed(retired[0], retired[1], now)

    def _retire(self, session: _Session, now: float) -> None:
        """`session` adds nothing to the sessions that share its blocks, or a
        later turn supersedes it: what only it owned goes first."""
        if self.sessions.pop(session.key, None) is None:
            return
        self._disown(session)
        self.stats.retired += 1
        self._retired[session.key] = (
            self._freq(session, now),
            now,
            session.gap_tool,
            session.gap_user,
        )
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
        """Its resumes, halved every half-life; with turn gaps on record
        (`abandon_gap_multiplier`), undecayed while it is not overdue: a slow
        agent's steady rhythm is no reason to forget it (window 5: 7 of the
        9 sessions re-read whole were slow agents)."""
        if self._patience(session) is not None and not self._overdue(session, now):
            return min(session.freq, _MAX_LIVE_FREQ)
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
        page_group_id = self.page_group_id
        reach = shared = previous = 0
        broken = False
        walk = _Walk()
        # Per unit, ascending: (position, whether it is exclusive, its blocks
        # in either tier, its RAM blocks, whether the GPU alone holds its
        # page, whether the GPU alone holds one of its blocks).
        units: list[tuple[int, bool, int, list[KVCacheBlock], bool, bool]] = []
        # Per point, ascending: (its index in `units`, its unit, its state
        # groups' blocks in either tier, its exclusive RAM state blocks).
        found: list[tuple[int, BlockHash, int, list[KVCacheBlock]]] = []
        for position, unit in reversed(session.units):
            exclusive = self.owners.get(unit) == {key}
            held = states = 0
            ram: list[KVCacheBlock] = []
            ram_states: list[KVCacheBlock] = []
            complete = True
            page_on_gpu_only = any_on_gpu_only = False
            for group_id in self.group_ids:
                grouped = make_block_hash_with_group_id(unit, group_id)
                blocks = cpu.get_all_blocks(grouped)
                if gone and blocks:
                    blocks = [b for b in blocks if b.block_id not in gone]
                if exclusive:
                    ram.extend(blocks)
                    if group_id == page_group_id:
                        walk.ram_pages += len(blocks)
                    else:
                        ram_states.extend(blocks)
                        walk.ram_states += len(blocks)
                gpu_only = not blocks and on_gpu(grouped)
                if blocks or gpu_only:
                    held += 1
                    states += group_id != page_group_id
                    if gpu_only:
                        any_on_gpu_only = True
                        page_on_gpu_only |= group_id == page_group_id
                elif group_id == page_group_id and not position % block_size:
                    # The chain has a hole: nothing past it resumes.
                    broken = True
                else:
                    complete = False
            units.append(
                (position, exclusive, held, ram, page_on_gpu_only, any_on_gpu_only)
            )
            if broken or not complete:
                continue
            found.append((len(units) - 1, unit, states, ram_states))
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
        if session.rungs and reach > shared:
            # Only ladder points are valued by point (`_rung_values`).
            self._add_points(session, walk, units, found, shared)
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
                # Not a queued request's: it may resume past this session's
                # reach (a preempted turn, its own point no session's yet).
                # Once its keys are found again they are no longer listed.
                walk.dead.extend(b for b in unlisted if not self._is_queued(b))
            elif last and position > cut and unlisted:
                walk.live.append(unlisted)
        walk.reach, walk.shared = reach, shared
        if reach > shared:
            walk.gain = self.cost.seconds(shared, reach)
            walk.gain_last = self.cost.seconds(max(previous, shared), reach)
        return walk

    def _add_points(
        self,
        session: _Session,
        walk: _Walk,
        units: list[tuple[int, bool, int, list[KVCacheBlock], bool, bool]],
        found: list[tuple[int, BlockHash, int, list[KVCacheBlock]]],
        shared: int,
    ) -> None:
        """`walk.points`: the session's points above `shared`, each with the
        segment under it (from the point below): the seconds it saves and its
        exclusive blocks in both tiers, as `_walk` counts the last segment."""
        points = walk.points
        by_index = {i: (unit, states, ram) for i, unit, states, ram in found}
        lower, size = shared, 0
        for i, (position, exclusive, held, *_) in enumerate(units):
            if position <= shared:
                continue
            if exclusive:
                size += held
            point = by_index.get(i)
            if point is None:
                continue
            unit, states, ram_states = point
            if points and points[-1].position == position:
                # Another unit ending at the same token: the same point, which
                # one unit's states going would not remove.
                points[-1].size += size
                points[-1].rung = False
            else:
                points.append(
                    _Point(
                        position,
                        unit,
                        self.cost.seconds(lower, position),
                        size,
                        unit in session.rungs,
                        states,
                        ram_states,
                    )
                )
            lower, size = position, 0

    def _score(
        self,
        session: _Session,
        now: float,
        gone: Collection[int] = (),
        skip: Collection[BlockHash] = (),
    ) -> float:
        return self._scored(session, now, gone, skip)[0]

    def _scored(
        self,
        session: _Session,
        now: float,
        gone: Collection[int] = (),
        skip: Collection[BlockHash] = (),
    ) -> tuple[float, _Walk, _Point | None]:
        """What evicting `session`'s last segment loses per block, its walk,
        and None; or, if one of its ladder points (`skip` aside) loses less
        per block, that and the point: a session gives up its cheap tail (a
        turn back to its previous resume point) long before the prefix under
        its first resume point, which only the whole prefill brings back."""
        walk = self._walk(session, gone)
        if walk.dead:
            # RAM blocks nothing reads: before any value.
            return _DEAD_SCORE, walk, None
        if walk.gain <= 0.0:
            self._to_retire.append(session)
            return float("inf"), walk, None
        scale = 1.0 + self._freq(session, now)
        overdue = self._overdue(session, now)

        def score_of(per_block: float) -> float:
            value = scale * per_block
            if overdue:
                # Abandoned by its own rhythm: before every live session, the
                # cheapest first.
                return _OVERDUE_SCORE + value / (1.0 + value)
            return session.stamp + value

        # No tail in RAM to give: infinite.
        score = (
            score_of(walk.gain_last / max(1, walk.size_last))
            if walk.live
            else float("inf")
        )
        choice = None
        for point, seconds in self._rung_values(session, walk, scale, overdue, skip):
            rung_score = score_of(seconds / max(1, point.states))
            if rung_score < score:
                score, choice = rung_score, point
        return score, walk, choice

    def _push(self, session: _Session) -> _Walk:
        session.version = next(self._seq)
        score, walk, _ = self._scored(session, self.now())
        if session.rungs and walk.points:
            self._drop_safe(session, walk)
        # A session with nothing exclusive to give is not pushed; it is again
        # when RAM stores one of its units or it loses a sharer.
        if score != float("inf"):
            heapq.heappush(self.heap, (score, session.version, session.key))
            self._unranked.discard(session.key)
        elif walk.gain > 0.0 and not walk.dead:
            self._unranked.add(session.key)
        return walk

    def _may_drop(self, session: BlockHash, unit: BlockHash) -> bool:
        return self.may_drop_point is None or self.may_drop_point(session, unit)

    def _drop_safe(self, session: _Session, walk: _Walk) -> None:
        """The ladder points no cut could stop at go at once (`_safe_drops`):
        the connector drops their states from both tiers
        (`take_dropped_points`); the session is pushed again then."""
        candidates = [
            p.position
            for p in walk.points[:-1]
            if p.rung and self._may_drop(session.key, p.unit)
        ]
        if not candidates:
            return
        units = {p.position: p.unit for p in walk.points}
        for position in _safe_drops(_segments(walk.points), candidates):
            unit = units[position]
            del session.rungs[unit]
            self._dropped_points.append((session.key, unit, "safe"))
            self.stats.rungs_safe += 1

    def _rung_values(
        self,
        session: _Session,
        walk: _Walk,
        scale: float,
        overdue: bool,
        skip: Collection[BlockHash],
    ) -> Iterator[tuple[_Point, float]]:
        """`session`'s ladder points RAM can drop the states of (`skip`
        aside), each with the seconds it is worth: with "expected" (and the
        clock's rise known, the session live) what losing it adds to the
        session's expected recompute, else the seconds its segment saves (as
        if its tail were sure to be cut, and the cut to stop at it)."""
        points = walk.points
        candidates = [
            k
            for k, p in enumerate(points[:-1])
            if p.rung
            and p.ram_states
            and p.unit not in skip
            and self._may_drop(session.key, p.unit)
        ]
        if not candidates:
            return
        rate = self.clock_rate
        if self.ladder_ranking != "expected" or overdue or rate is None:
            for k in candidates:
                yield points[k], points[k].gain
            return
        segments = _segments(points)
        level, chance = self._cut_model(session, scale, rate)
        base = _expected_recompute(segments, level, chance)
        for k in candidates:
            without = _expected_recompute(_merged(segments, k), level, chance)
            yield points[k], max(0.0, without - base)

    def _cut_model(
        self, session: _Session, scale: float, rate: float
    ) -> tuple[Callable[[float, int], float], Callable[[float], float]]:
        """The clock at which a segment of `session` goes (`seconds` saved
        over `blocks`), and the chance the clock gets there before the
        session's next turn: its return and the clock's rise race, each at a
        steady rate (a memoryless race: gap / (gap + time to reach it))."""
        stamp, clock = session.stamp, self.clock
        gap = self._expected_gap(session)

        def level(seconds: float, blocks: int) -> float:
            return stamp + scale * seconds / max(1, blocks)

        def chance(at: float) -> float:
            if at <= clock:
                return 1.0
            return gap / (gap + (at - clock) / rate)

        return level, chance

    def _chance_tail_cut(
        self, session: _Session, walk: _Walk, now: float
    ) -> float | None:
        """The chance `session`'s last segment is cut before its next turn
        (`_cut_model`), or None: the clock's rise unknown, the session
        overdue or with no tail of its own."""
        rate = self.clock_rate
        if rate is None or walk.gain <= 0.0 or self._overdue(session, now):
            return None
        scale = 1.0 + self._freq(session, now)
        level, chance = self._cut_model(session, scale, rate)
        return chance(level(walk.gain_last, walk.size_last))

    def take_dropped_points(self) -> list[tuple[BlockHash, BlockHash, str]]:
        """The ladder points chosen to go since the last call, (session, hash
        unit, "safe" | "ranked"): the connector drops their states from both
        tiers and pushes the sessions again (`repush`). A ranked one's RAM
        states are victims already."""
        dropped, self._dropped_points = self._dropped_points, []
        return dropped

    def repush(self, keys: Iterable[BlockHash]) -> None:
        """Rank sessions `keys` again: their blocks changed outside an
        eviction (a ladder point's states dropped). The push makes their
        overdue check (`_due`) stale, so it is scheduled again."""
        for key in set(keys):
            session = self.sessions.get(key)
            if session is not None:
                self._push(session)
                self._schedule_due(session)

    def unranked_blocks(self) -> tuple[int, int]:
        """Live sessions' exclusive RAM blocks no eviction ranks (pages,
        states): a session whose tail the GPU holds has no tail to give by
        value (only its ladder points, if ranked), and RAM takes its other
        blocks only in its own LRU order. Walks only the sessions last scored
        so (`_unranked`)."""
        now = self.now()
        pages = states = 0
        for key in list(self._unranked):
            session = self.sessions.get(key)
            if session is None:
                self._unranked.discard(key)
                continue
            walk = self._walk(session)
            if walk.dead or walk.gain <= 0.0 or walk.live:
                continue
            ranked = sum(
                len(point.ram_states)
                for point, _ in self._rung_values(
                    session,
                    walk,
                    1.0 + self._freq(session, now),
                    self._overdue(session, now),
                    (),
                )
            )
            pages += walk.ram_pages
            states += walk.ram_states - ranked
        return pages, states

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
        (`_take_queued`). Any shortfall goes in the pool's LRU order, live
        sessions' blocks included (`RamEvictionStats.unranked`). Copies whose
        GPU blocks are about to leave (`leaving`) go only once nothing but
        queued requests' blocks is left: the GPU would copy such a block
        again, evicting another such copy, while the allocations held for
        them wait on. When one goes all the same, its GPU block may leave
        uncopied (`dropped`): RAM's value order chose it."""
        self._decay()
        self._sample_clock(self.now())
        self._rescore_overdue(self.now())
        queue = self.pool.free_block_queue
        victims: list[KVCacheBlock] = []
        seen: set[int] = set()
        fresh = self._fresh | self._fresher
        was_fresh = set(fresh)
        self._fresher, self._fresh = self._fresh, set()
        leaving: set[int] = (
            {b.block_id for b in self.leaving()} if self.leaving is not None else set()
        )
        was_leaving = set(leaving)
        # The kind of eviction under way, by victim (`_note_taken`).
        kind = ["unkeyed"]
        kinds: dict[int, str] = {}

        def may_go(
            block: KVCacheBlock, even_fresh: bool = False, even_queued: bool = False
        ) -> bool:
            return not (
                block.ref_cnt
                or block.block_id in seen
                or block.is_null
                or (block.block_id in fresh and not even_fresh)
                or (not even_queued and self._is_queued(block))
                or block.block_id in leaving
            )

        def take(
            block: KVCacheBlock,
            even_fresh: bool = False,
            even_queued: bool = False,
            label: str | None = None,
        ) -> bool:
            """Takes `block` if it may go."""
            if not may_go(block, even_fresh, even_queued):
                return False
            seen.add(block.block_id)
            victims.append(block)
            kinds[block.block_id] = label or kind[0]
            return True

        extra = self.pool.cached_block_hashes_by_block

        def take_orphans() -> bool:
            """Blocks no session owns, in LRU order; fresh ones too if only
            retired sessions owned them (the GPU does not copy them again)."""
            kind[0] = "orphan"
            for block in queue.iter_blocks_after(None):
                if block.block_hash is None or self._owned(block):
                    continue
                keys = (block.block_hash, *extra.get(block.block_id, ()))
                if take(block, self.disowned(keys)):
                    self.stats.orphans += 1
                    if len(victims) >= num_blocks:
                        return True
            return False

        def take_ranked(label: str = "session") -> None:
            if not take_orphans():
                kind[0] = "duplicate"
                self._take_duplicates(take, victims, num_blocks)
                if len(victims) < num_blocks:
                    kind[0] = label
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
            if len(victims) < num_blocks and leaving:
                # Copies of blocks about to leave too, by value, before what
                # queued requests resume from.
                leaving.clear()
                take_ranked("leaving")
            if len(victims) < num_blocks and self._queued:
                kind[0] = "queued"
                self._take_queued(take, victims, num_blocks)
            if len(victims) < num_blocks:
                # The rest in the pool's LRU order, live sessions' blocks too.
                kind[0] = "unranked"
                for block in queue.iter_blocks_after(None):
                    if take(block, True, True):
                        self.stats.unranked += 1
                        if len(victims) >= num_blocks:
                            break
        if was_leaving and (taken := [b for b in victims if b.block_id in was_leaving]):
            self.stats.leaving += len(taken)
            # A copy of a block the GPU keeps (a duplicate) drops nothing: the
            # walk copies that block again once it is about to leave.
            keeps = self.gpu_keeps() if self.gpu_keeps is not None else None
            dropped = [
                b
                for b in taken
                if keeps is None
                or not all(
                    keeps(key) for key in (b.block_hash, *extra.get(b.block_id, ()))
                )
            ]
            if dropped and self.dropped is not None:
                self.dropped(dropped)
        self.stats.rung_fresh += sum(
            kinds[b.block_id] == "rung" and b.block_id in was_fresh for b in victims
        )
        # The pool takes the first `num_blocks` (a unit can take it past).
        self._note_taken(victims[:num_blocks], kinds)
        moved = [b for b in victims if b.prev_free_block is not None]
        for block in moved:
            queue.remove(block)
        queue.prepend_n(moved)
        self._retire_pending(self.now())

    def _note_taken(self, taken: list[KVCacheBlock], kinds: dict[int, str]) -> None:
        """Before the pool evicts `taken`: note the point each session owning
        some of them resumes at, while they are still cached, for `settle`."""
        extra = self.pool.cached_block_hashes_by_block
        for block in taken:
            if block.block_hash is None:
                continue
            self.note_evicting(
                (block.block_hash, *extra.get(block.block_id, ())),
                kinds.get(block.block_id, "unranked"),
            )

    def note_evicting(self, keys: Iterable[BlockHashWithGroupId], kind: str) -> None:
        """A block under `keys` is about to leave a tier with no copy in the
        other (`kind`: how): note the point each session owning one of them
        resumes at, while it is still cached, for `settle`. The GPU dropping
        a block RAM does not hold loses a session's point just as RAM
        evicting one does."""
        owners: set[BlockHash] = set()
        for key in keys:
            owners.update(self.owners.get(get_block_hash(key), ()))
        for owner in owners:
            touched = self._touched.get(owner)
            if touched is None:
                session = self.sessions.get(owner)
                if session is None:
                    continue
                touched = (session, self._walk(session).reach, [0], set())
                self._touched[owner] = touched
            touched[2][0] += 1
            touched[3].add(kind)

    def settle(self) -> list[SessionLoss]:
        """After the evictions since the last call: the sessions they left
        with no point of their own to resume at, which resumed deeper than
        what they share before (`RamEvictionStats.sessions_lost`). A session
        loses its tail first, so this is the eviction that took its last
        resume point: its next turn re-reads it whole."""
        if not self._touched:
            return []
        now = self.now()
        losses = []
        for key, (session, before, blocks, kinds) in self._touched.items():
            walk = self._walk(session)
            if before > walk.shared and walk.reach <= walk.shared:
                self.stats.sessions_lost += 1
                losses.append(
                    SessionLoss(
                        key,
                        session.end,
                        before,
                        walk.reach,
                        blocks[0],
                        tuple(sorted(kinds)),
                        now - session.freq_at,
                    )
                )
        self._touched.clear()
        return losses

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
        while one may not go yet (in use or fresh) would lose it unfreed. A
        ladder point worth less goes before the segment (`_scored`): its
        state blocks all or none, its page staying, so that the segments
        around it join."""
        heap = self.heap
        now = self.now()
        # A session that gives nothing (its blocks in use or fresh) is pushed
        # back only after this eviction: it would come straight back.
        visited: list[_Session] = []
        # By session, the ladder points that may not go in this eviction: it
        # is ranked again without them.
        skipped: dict[BlockHash, set[BlockHash]] = {}
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
            skip = skipped.get(key, ())
            current, walk, choice = self._scored(session, now, gone, skip)
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
            if choice is not None and not dead:
                blocks = choice.ram_states
                if not (
                    all(may_go(block) for block in blocks)
                    and self._may_drop(key, choice.unit)
                ):
                    skipped.setdefault(key, set()).add(choice.unit)
                else:
                    for block in blocks:
                        take(block, label="rung")
                    del session.rungs[choice.unit]
                    self._dropped_points.append((key, choice.unit, "ranked"))
                    self.stats.rungs += 1
                    self.stats.rung_blocks += len(victims) - taken
                    if current >= 0.0:
                        self.clock = max(self.clock, current)
                session.version = next(self._seq)
                rest = self._score(session, now, gone, skipped.get(key, ()))
                if rest != float("inf"):
                    heapq.heappush(heap, (rest, session.version, key))
                if key in skipped or rest == float("inf"):
                    # Ranked again after this eviction, with every rung: one
                    # refused now may go then, and a session left with nothing
                    # to give now would leave the heap.
                    visited.append(session)
                continue
            if dead:
                for block in walk.dead:
                    if take(block) and len(victims) >= num_blocks:
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
                if current >= 0.0:
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

    def _sample_clock(self, now: float) -> None:
        """Measure the clock's rise per second (`clock_rate`), which values
        ladder points with "expected"; their scores are rebuilt when it
        changes much, or becomes known or unknown."""
        samples = self._clock_samples
        if not samples or now - samples[-1][0] >= 1.0:
            samples.append((now, self.clock))
        while len(samples) > 1 and now - samples[1][0] >= _RATE_WINDOW_S:
            samples.popleft()
        if len(samples) > 1 and now - samples[0][0] >= 2 * _RATE_WINDOW_S:
            # No eviction for a while: the rate is the burst's, not the gap's.
            samples.popleft()
        first_at, first = samples[0]
        span = now - first_at
        rate = (
            (self.clock - first) / span
            if span >= _RATE_MIN_SPAN_S and self.clock > first
            else None
        )
        self.clock_rate = rate
        built = self._rate_built
        if self.ladder_ranking != "expected" or rate == built:
            return
        if (
            rate is None
            or built is None
            or max(rate / built, built / rate) > _RATE_REBUILD_RATIO
        ):
            self._rate_built = rate
            self._rebuild(now)

    def _decay(self) -> None:
        now = self.now()
        if now - self._decayed_at < self._decay_every:
            return
        self._rebuild(now)

    def _rebuild(self, now: float) -> None:
        self._decayed_at = now
        # Frequencies decay continuously; scores are rebuilt at these steps.
        heap = []
        self._unranked.clear()
        for session in self.sessions.values():
            session.version = next(self._seq)
            score, walk, _ = self._scored(session, now)
            if score != float("inf"):
                heap.append((score, session.version, session.key))
            elif walk.gain > 0.0 and not walk.dead:
                self._unranked.add(session.key)
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
        self._touched.clear()
        self._due.clear()
        self._dropped_points.clear()
        self._clock_samples.clear()
        self.clock_rate = self._rate_built = None
        self._unranked.clear()
