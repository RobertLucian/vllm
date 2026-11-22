# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TieredCPUOffloadConnector on the spec-decode device harness: a RAM tier
that extends a small GPU pool resumes every session where a GPU that never
evicts would, its exact end included. The device checks every state a
request reads and every state cached or restored."""

import itertools
import random

import pytest

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_group_id,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.request import Request
from vllm.v1.simple_kv_offload.tiered import _REQUEUE_STEPS, TieredCPUOffloadScheduler

from .spec_device import Device, Engine
from .test_speculative_frontiers import (
    DRAFTS,
    assert_clean,
    default_accept,
    speculative_config,
)
from .utils import make_hybrid_scheduler, prompt, tiered_cpu_connector

pytestmark = pytest.mark.cpu_test

BLOCK = 64
# Whole-block hashing, and hashing finer than the block (production hashes
# every 32 tokens of 1600-token blocks).
HASH_BLOCK_SIZES = [BLOCK, BLOCK // 4]


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def make_engine(
    async_scheduling: bool,
    hash_block_size: int,
    *,
    num_blocks: int = 90,
    num_cpu_blocks: int | None = 400,
    ram_eviction: str = "lru",
    ram_state_spacing_blocks: int = 8,
    copy_ahead_blocks: int = 0,
) -> Engine:
    """The tiered connector with a RAM pool of `num_cpu_blocks`, or (None) no
    connector."""
    scheduler = make_hybrid_scheduler(
        block_size=BLOCK,
        num_blocks=num_blocks,
        max_num_batched_tokens=256,
        ring=0,
        # The scheduler's hash unit, and the connector's (`prefix_match_unit`
        # is what production sets).
        hash_block_size=hash_block_size,
        prefix_match_unit=None if hash_block_size == BLOCK else hash_block_size,
        speculative_config=speculative_config(),
        num_speculative_blocks=DRAFTS,
        async_scheduling=async_scheduling,
        mamba_frontier_checkpoint=True,
        kv_transfer_config=(
            None
            if num_cpu_blocks is None
            else tiered_cpu_connector(
                num_cpu_blocks,
                ram_eviction,
                ram_state_spacing_blocks,
                copy_ahead_blocks,
            )
        ),
    )
    device = Device(scheduler, num_drafts=DRAFTS, truncate=True, accept=default_accept)
    return Engine(scheduler, device, async_scheduling)


def roomy_engine(async_scheduling: bool, hash_block_size: int) -> Engine:
    """A GPU pool every session fits in: nothing is ever evicted."""
    return make_engine(
        async_scheduling, hash_block_size, num_blocks=2000, num_cpu_blocks=None
    )


def manager(engine: Engine) -> TieredCPUOffloadScheduler:
    connector = engine.scheduler.connector
    assert connector is not None
    manager = connector.scheduler_manager  # type: ignore[attr-defined]
    assert isinstance(manager, TieredCPUOffloadScheduler)
    return manager


def assert_nothing_lost(engine: Engine) -> None:
    assert_clean(engine)
    assert manager(engine).lazy_spill_stats.unspilled_evicted_blocks == 0


def rotate(
    engine: Engine, *, seed: int = 5, sessions: int = 6, turns: int = 3
) -> list[int]:
    """Sessions one after another, `turns` turns each; returns the computed
    tokens every later turn was admitted with."""
    rng = random.Random(seed)
    tokens = {s: prompt(s, rng.randrange(1000, 1400)) for s in range(sessions)}
    resumed = []
    for turn in range(turns):
        for s in range(sessions):
            request_id = f"r{s}t{turn}"
            engine.add(request_id, tokens[s], rng.choice((17, 90)))
            engine.run(request_id)
            if turn:
                resumed.append(engine.admissions[request_id][-1])
            tokens[s] = list(engine.requests[request_id].all_token_ids) + prompt(
                2000 + 10 * s + turn, rng.randrange(10, 90)
            )
    return resumed


@pytest.mark.parametrize("hash_block_size", HASH_BLOCK_SIZES)
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_sessions_restored_from_ram_resume_where_a_gpu_that_never_evicts_does(
    async_scheduling: bool, hash_block_size: int
):
    """Six sessions rotate through a GPU pool that holds about three. Each
    later turn resumes from RAM at the same point as on a GPU that keeps
    every session: the previous turn's exact end."""
    engine = make_engine(async_scheduling, hash_block_size)
    resumed = rotate(engine)
    assert_nothing_lost(engine)
    stats = manager(engine).restore_stats
    assert stats.completed > 0 and stats.restored_frontiers > 0, stats
    expected = rotate(roomy_engine(async_scheduling, hash_block_size))
    assert resumed == expected, (resumed, expected)


@pytest.mark.parametrize("hash_block_size", HASH_BLOCK_SIZES)
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_concurrent_resumes_restore_whole_sessions(
    async_scheduling: bool, hash_block_size: int
):
    """Six sessions overflow the GPU pool and resume at once, in an order
    that splits some across the tiers. Every resume lands where it would on
    a GPU that keeps every session."""

    def resume_all(engine: Engine) -> list[int]:
        for s in range(8):
            engine.add(f"a{s}", prompt(100 + s, 1300), 17)
            engine.run(f"a{s}")
        tokens = {}
        for pair in range(3):
            built = []
            for s in (2 * pair, 2 * pair + 1):
                engine.add(f"b{s}", prompt(300 + s, 1100), 17 if s % 2 else 40)
                built.append(f"b{s}")
            engine.run(*built)
            for s in (2 * pair, 2 * pair + 1):
                tokens[s] = list(engine.requests[f"b{s}"].all_token_ids) + prompt(
                    400 + s, 40
                )
        resumed = []
        for s in (3, 1, 4, 0, 5, 2):
            engine.add(f"b{s}r", tokens[s], 17)
            resumed.append(f"b{s}r")
        engine.run(*resumed)
        assert_clean(engine)
        return [engine.admissions[f"b{s}r"][-1] for s in range(6)]

    tiered = make_engine(async_scheduling, hash_block_size)
    resumed = resume_all(tiered)
    assert_nothing_lost(tiered)
    expected = resume_all(roomy_engine(async_scheduling, hash_block_size))
    assert resumed == expected, (resumed, expected)


def _split_session(engine: Engine, *, drop_page: int | None) -> int:
    """A session restored from RAM, then (`drop_page`) one of its middle
    pages dropped from the GPU while RAM keeps it: GPU, RAM-only, GPU, then
    the GPU's frontier. Returns the next turn's admission."""
    engine.add("s0", prompt(1, 1300), 17)
    engine.run("s0")
    for s in range(6):
        engine.add(f"f{s}", prompt(50 + s, 1300), 17)
        engine.run(f"f{s}")
    tokens = list(engine.requests["s0"].all_token_ids) + prompt(900, 40)
    engine.add("s1", tokens, 17)
    engine.run("s1")
    request = engine.requests["s1"]
    if drop_page is not None:
        scheduler = engine.scheduler
        pool = scheduler.kv_cache_manager.block_pool
        tier = manager(engine)
        per_block = BLOCK // scheduler.hash_block_size
        key = make_block_hash_with_group_id(
            request.block_hashes[(drop_page + 1) * per_block - 1], tier.fa_gidx
        )
        page = pool.cached_block_hash_to_block.get_one_block(key)
        assert page is not None and page.ref_cnt == 0
        assert tier.cpu_block_pool.cached_block_hash_to_block.get_one_block(key)
        pool.evict_blocks({page.block_id})
    tokens = list(request.all_token_ids) + prompt(901, 40)
    engine.add("s2", tokens, 17)
    engine.run("s2")
    return engine.admissions["s2"][-1]


@pytest.mark.parametrize("hash_block_size", HASH_BLOCK_SIZES)
def test_a_session_split_around_a_ram_only_page_resumes_whole(hash_block_size: int):
    """The GPU holds a session's pages but one in the middle, which only RAM
    holds: the restore loads that page and the turn resumes at the GPU's
    frontier, as on a GPU that keeps everything."""
    engine = make_engine(False, hash_block_size, num_cpu_blocks=2000)
    resumed = _split_session(engine, drop_page=8)
    assert_nothing_lost(engine)
    expected = _split_session(roomy_engine(False, hash_block_size), drop_page=None)
    assert resumed == expected, (resumed, expected)


def _changed_answer(engine: Engine) -> int:
    """A turn whose history carries a different previous answer (a client
    that re-renders it): no exact end matches."""
    tokens = prompt(7, 1200)
    engine.add("c0", tokens, 60)
    engine.run("c0")
    for s in range(6):
        engine.add(f"g{s}", prompt(70 + s, 1300), 17)
        engine.run(f"g{s}")
    answer = list(engine.requests["c0"].output_token_ids)
    changed = tokens + [t + 1 for t in answer] + prompt(950, 40)
    engine.add("c1", changed, 17)
    engine.run("c1")
    return engine.admissions["c1"][-1]


@pytest.mark.parametrize("hash_block_size", HASH_BLOCK_SIZES)
def test_a_changed_answer_resumes_at_the_deepest_shared_point(hash_block_size: int):
    """When the previous answer comes back changed, the turn resumes at the
    deepest point both share (the prompt's tail state), as on a GPU that
    keeps everything."""
    engine = make_engine(False, hash_block_size, num_cpu_blocks=2000)
    resumed = _changed_answer(engine)
    assert_nothing_lost(engine)
    expected = _changed_answer(roomy_engine(False, hash_block_size))
    assert resumed == expected, (resumed, expected)


def _fuzz_case(engine: Engine, seed: int) -> list[int]:
    rng = random.Random(seed)
    num_sessions = rng.randrange(4, 8)
    tokens = {
        s: prompt(seed * 100 + s, rng.randrange(600, 1300)) for s in range(num_sessions)
    }
    resumed = []
    for turn in range(3):
        order = list(range(num_sessions))
        rng.shuffle(order)
        batch = rng.choice((1, 2, num_sessions))
        for start in range(0, num_sessions, batch):
            ids = []
            for s in order[start : start + batch]:
                request_id = f"z{s}t{turn}"
                engine.add(request_id, tokens[s], rng.choice((9, 33, 70)))
                ids.append(request_id)
            engine.run(*ids)
            for request_id in ids:
                s = int(request_id[1 : request_id.index("t")])
                if turn:
                    # Where it resumed, not where a preemption re-admitted it.
                    resumed.append(engine.admissions[request_id][0])
                tokens[s] = list(engine.requests[request_id].all_token_ids) + prompt(
                    seed * 1000 + 10 * s + turn, rng.randrange(8, 80)
                )
    return resumed


@pytest.mark.parametrize("hash_block_size", HASH_BLOCK_SIZES)
@pytest.mark.parametrize("seed", range(4))
def test_random_sessions_resume_where_a_gpu_that_never_evicts_does(
    seed: int, hash_block_size: int
):
    """Random sessions, turn lengths, orders and concurrency through a small
    GPU pool and a RAM pool that keeps everything."""
    engine = make_engine(seed % 2 == 0, hash_block_size, num_cpu_blocks=2000)
    resumed = _fuzz_case(engine, seed)
    assert_nothing_lost(engine)
    expected = _fuzz_case(roomy_engine(seed % 2 == 0, hash_block_size), seed)
    assert resumed == expected, (resumed, expected)


def test_resumes_are_measured_by_tier():
    """The connector counts the tokens a resume recomputes that a finished
    request had processed: none for exact-end resumes, from RAM or GPU."""
    engine = make_engine(False, BLOCK)
    rotate(engine)
    stats = manager(engine).resume_stats
    assert stats.counts.get("ram", 0) > 0, stats
    assert stats.tokens.get("ram", 0) == 0, stats
    assert stats.buckets["ram", "0"] == stats.counts["ram"]


def _boundary_states(engine: Engine, request_id: str) -> list[int]:
    """The block boundaries of `request_id`'s chain whose state the GPU holds
    in every Mamba group."""
    scheduler = engine.scheduler
    request = engine.requests[request_id]
    tier = manager(engine) if scheduler.connector is not None else None
    checkpoints = scheduler.kv_cache_manager.checkpoints
    assert checkpoints is not None
    per_block = BLOCK // scheduler.hash_block_size
    held = []
    for i in range(1, len(request.block_hashes) // per_block + 1):
        key = request.block_hashes[i * per_block - 1]
        if checkpoints.complete(key):
            held.append(i)
    assert tier is None or tier.num_gpu_states_pruned >= 0
    return held


@pytest.mark.parametrize("hash_block_size", HASH_BLOCK_SIZES)
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_a_finished_session_keeps_only_its_resume_states_on_the_gpu(
    async_scheduling: bool, hash_block_size: int
):
    """The GPU holds a state at the block boundaries its prefill chunks ended
    at. Once the request finishes, it keeps those RAM would keep (every 8th
    block, the last boundaries before its prompt end and its end) and
    releases the rest; a GPU without the tier keeps them all. The next turn
    still resumes at the exact end."""
    engine = make_engine(async_scheduling, hash_block_size, num_blocks=400)
    roomy = roomy_engine(async_scheduling, hash_block_size)
    tokens = prompt(11, 1500)
    for e in (engine, roomy):
        e.add("p0", tokens, 40)
        e.run("p0")
    prompt_floor, end_floor = 1500 // BLOCK, (1500 + 39) // BLOCK
    # Finer hashing caches the prompt tail inside the last prompt block
    # instead of a state at its boundary.
    held = set(_boundary_states(roomy, "p0"))
    kept = {8, 16, prompt_floor, end_floor} & held
    assert _boundary_states(engine, "p0") == sorted(kept)
    assert held - kept
    tier = manager(engine)
    assert tier.num_gpu_states_pruned > 0
    assert tier.prune_outcomes["pruned"] == len(held - kept)
    nxt = list(engine.requests["p0"].all_token_ids) + prompt(12, 40)
    for e in (engine, roomy):
        e.add("p1", nxt, 17)
        e.run("p1")
    assert engine.admissions["p1"][0] == roomy.admissions["p1"][0]
    assert_nothing_lost(engine)


def test_a_turn_that_diverges_mid_session_resumes_at_the_nearest_kept_state():
    """A turn that parts from the session 1,300 tokens in (an edited earlier
    message) resumes at the last kept state before that, 16 blocks in, where
    a GPU that keeps every state resumes at the last one it cached before."""
    engine = make_engine(False, BLOCK, num_blocks=400)
    roomy = roomy_engine(False, BLOCK)
    tokens = prompt(13, 1500)
    for e in (engine, roomy):
        e.add("d0", tokens, 40)
        e.run("d0")
    before = max(b for b in _boundary_states(roomy, "d0") if b * BLOCK <= 1300)
    for e in (engine, roomy):
        e.add("d1", tokens[:1300] + prompt(14, 60), 17)
        e.run("d1")
    assert roomy.admissions["d1"][0] == before * BLOCK > 16 * BLOCK
    assert engine.admissions["d1"][0] == 16 * BLOCK
    assert_nothing_lost(engine)


def _ram_states(engine: Engine, request_id: str) -> list[int]:
    """The block boundaries of `request_id`'s chain whose state RAM holds in
    some Mamba group."""
    tier = manager(engine)
    request = engine.requests[request_id]
    per_block = BLOCK // engine.scheduler.hash_block_size
    cpu = tier.cpu_block_pool.cached_block_hash_to_block
    held = []
    for i in range(1, len(request.block_hashes) // per_block + 1):
        key = request.block_hashes[i * per_block - 1]
        if any(
            cpu.get_one_block(make_block_hash_with_group_id(key, group_id))
            for group_id in tier.mamba_gids
        ):
            held.append(i)
    return held


@pytest.mark.parametrize("hash_block_size", HASH_BLOCK_SIZES)
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_a_tool_loop_keeps_only_its_latest_turn_and_loop_start_points(
    async_scheduling: bool, hash_block_size: int
):
    """Each turn of a tool loop extends the previous turn's end. The GPU then
    keeps the states at the last boundaries before the latest turn's prompt
    end and end and before the loop's first prompt end, and no others: the
    points earlier turns kept go once the next turn finishes."""
    engine = make_engine(
        async_scheduling, hash_block_size, num_blocks=400, ram_state_spacing_blocks=0
    )
    roomy = roomy_engine(async_scheduling, hash_block_size)
    tokens = prompt(21, 700)
    for turn in range(5):
        request_id = f"l{turn}"
        for e in (engine, roomy):
            e.add(request_id, tokens, 40)
            e.run(request_id)
        assert engine.admissions[request_id][0] == roomy.admissions[request_id][0]
        tokens = list(engine.requests[request_id].all_token_ids) + prompt(
            30 + turn, 150
        )
    request = engine.requests[request_id]
    end = len(request.all_token_ids) - 1
    points = {700 // BLOCK, request.num_prompt_tokens // BLOCK, end // BLOCK}
    held = set(_boundary_states(roomy, request_id))
    assert set(_boundary_states(engine, request_id)) == points & held
    assert manager(engine).prune_outcomes["pruned"] > 0
    assert_nothing_lost(engine)


@pytest.mark.parametrize(
    ("ram_eviction", "num_cpu_blocks"), [("gdsf", 48), ("lru", 400)]
)
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_a_user_prompt_after_a_tool_loop_resumes_at_the_loop_start(
    async_scheduling: bool, ram_eviction: str, num_cpu_blocks: int
):
    """A reasoning model's template drops a tool loop's reasoning once a user
    prompt follows, so that turn parts from its session a few tokens before
    the end of the prompt that began the loop. Two sessions take turns
    through a GPU too small for both, loop after loop. With room in RAM every
    turn resumes as deep as on a GPU that never evicts; with RAM full under
    GDSF none resumes more than a block short of that (the policy gives up
    the cheapest tail, an exact end, first). RAM holds no state where
    neither session resumes."""
    slack = 0 if ram_eviction == "lru" else BLOCK
    hash_block_size = BLOCK // 4
    engine = make_engine(
        async_scheduling,
        hash_block_size,
        num_blocks=36,
        num_cpu_blocks=num_cpu_blocks,
        ram_eviction=ram_eviction,
        ram_state_spacing_blocks=0,
    )
    roomy = roomy_engine(async_scheduling, hash_block_size)
    sessions = {s: prompt(40 + s, 400) for s in range(2)}
    loop_starts = {s: 400 for s in range(2)}
    latest: dict[int, str] = {}
    for loop in range(4):
        began = dict(loop_starts)
        for turn in range(3):
            for s in range(2):
                request_id = f"s{s}l{loop}t{turn}"
                for e in (engine, roomy):
                    e.add(request_id, sessions[s], 20)
                    e.run(request_id)
                short = (
                    roomy.admissions[request_id][0] - engine.admissions[request_id][0]
                )
                assert 0 <= short <= slack, (request_id, short)
                latest[s] = request_id
                # A tool turn: the reply stays and a tool result follows.
                seed = 1000 + 100 * s + 10 * loop + turn
                sessions[s] = list(engine.requests[request_id].all_token_ids) + prompt(
                    seed, 40
                )
        for s in range(2):
            # A user prompt: the loop renders again without its reasoning,
            # from 3 tokens before the end of the prompt that began it.
            seed = 5000 + 100 * s + 10 * loop
            sessions[s] = (
                sessions[s][: loop_starts[s] - 3]
                + prompt(seed, 60)
                + prompt(seed + 1, 50)
            )
            loop_starts[s] = len(sessions[s])
    tier = manager(engine)
    for s, request_id in latest.items():
        request = engine.requests[request_id]
        end = len(request.all_token_ids) - 1
        points = {began[s] // BLOCK, request.num_prompt_tokens // BLOCK, end // BLOCK}
        assert set(_ram_states(engine, request_id)) <= points, request_id
    assert tier.num_ram_states_dropped > 0
    assert_nothing_lost(engine)


def test_copy_ahead_keeps_the_eviction_end_copied():
    """With `copy_ahead_blocks` the walk keeps that many blocks at the GPU's
    eviction end copied (or droppable), so an allocation that takes them,
    a restore's most of all, never waits for a copy."""
    engine = make_engine(False, BLOCK, num_blocks=90, copy_ahead_blocks=40)
    rotate(engine)
    # An idle step lets the walk cover the blocks the last turns freed.
    engine.step()
    engine.step()
    tier = manager(engine)
    pool = engine.scheduler.kv_cache_manager.block_pool
    ahead = list(itertools.islice(pool.free_block_queue.iter_blocks_after(None), 40))
    assert len(ahead) == 40
    assert not [b.block_id for b in ahead if tier.needs_spill(b)]
    assert_nothing_lost(engine)


def test_states_the_graph_no_longer_tracks_are_pruned_from_the_gpu_cache():
    """The checkpoint graph lets a superseded boundary state go while the
    running request still holds its blocks; they stay cached on the GPU with
    no unit (production showed only such states). Once the request finishes,
    pruning evicts them from the GPU cache directly, keeping its resume
    points."""
    engine = make_engine(False, BLOCK, num_blocks=400, ram_state_spacing_blocks=0)
    roomy = roomy_engine(False, BLOCK)
    tier = manager(engine)
    tier.prune_gpu_states = False
    tokens = prompt(15, 1500)
    for e in (engine, roomy):
        e.add("u0", tokens, 40)
        e.run("u0")
    request = engine.requests["u0"]
    checkpoints = engine.scheduler.kv_cache_manager.checkpoints
    per_block = BLOCK // engine.scheduler.hash_block_size
    held = set(_boundary_states(engine, "u0"))
    assert held == set(_boundary_states(roomy, "u0"))
    points = {1500 // BLOCK, (1500 + 39) // BLOCK} & held
    stale = [request.block_hashes[i * per_block - 1] for i in sorted(held - points)]
    assert stale
    for key in stale:
        checkpoints.graph.removed(key)
    # The states are still cached on the GPU, in every Mamba group.
    assert set(_cached_states(engine, "u0")) == held
    tier.prune_gpu_states = True
    tier._prune_pending.extend(stale)
    engine.add("u1", prompt(16, 30), 4)
    engine.run("u1")
    assert set(_cached_states(engine, "u0")) == points
    assert tier.prune_outcomes["pruned untracked"] == len(stale)
    nxt = list(request.all_token_ids) + prompt(17, 40)
    for e in (engine, roomy):
        e.add("u2", nxt, 17)
        e.run("u2")
    assert engine.admissions["u2"][0] == roomy.admissions["u2"][0]
    assert_nothing_lost(engine)


def _cached_states(engine: Engine, request_id: str) -> list[int]:
    """The block boundaries of `request_id`'s chain whose state the GPU cache
    holds in every Mamba group, whether the checkpoint graph tracks it."""
    tier = manager(engine)
    request = engine.requests[request_id]
    per_block = BLOCK // engine.scheduler.hash_block_size
    gpu = engine.scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block
    return [
        i
        for i in range(1, len(request.block_hashes) // per_block + 1)
        if all(
            gpu.get_one_block(
                make_block_hash_with_group_id(
                    request.block_hashes[i * per_block - 1], group_id
                )
            )
            for group_id in tier.mamba_gids
        )
    ]


def test_ram_holds_each_session_once_when_its_blocks_go_back_to_the_gpu():
    """A session restored from RAM runs on the GPU and stays cached there; RAM
    still holds copies of its blocks. RAM drops those copies first (the GPU
    copies a block again once it is about to drop it), so a pool that holds
    every session once, GPU and RAM together, resumes them all with no full
    re-read. Mirroring them instead re-reads several sessions in full."""
    engine = make_engine(
        False,
        BLOCK,
        num_cpu_blocks=140,
        ram_eviction="gdsf",
        ram_state_spacing_blocks=0,
    )
    resumed = rotate(engine)
    expected = rotate(roomy_engine(False, BLOCK))
    assert not [r for r in resumed if r == 0]
    # At most the block before an end whose exact-end state GDSF gave up (its
    # cheapest tail): a block plus that turn's output (rotate's at most 90).
    assert max(e - r for r, e in zip(resumed, expected)) <= BLOCK + 90
    assert manager(engine).ram_policy.stats.duplicates > 0
    assert_nothing_lost(engine)


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_a_restore_that_waits_for_room_keeps_its_place_in_the_queue(
    async_scheduling: bool,
):
    """A turn whose session only RAM holds waits for GPU blocks while a long
    request holds most of them. It waits as long as that takes instead of
    resuming from the GPU alone (recomputing what RAM holds), and a request
    that arrives after it waits behind it until the restore has its blocks,
    instead of taking the blocks it waits for: as a load that waits for
    blocks makes the queue behind it wait."""
    engine = make_engine(async_scheduling, BLOCK, num_blocks=40)
    engine.add("s0", prompt(1, 1300), 17)
    engine.run("s0")
    for s in range(3):
        engine.add(f"f{s}", prompt(60 + s, 1300), 17)
        engine.run(f"f{s}")
    tier = manager(engine)
    request = engine.requests["s0"]
    engine.add("long", prompt(70, 1200), 500)
    while engine.requests["long"].num_output_tokens < 3:
        engine.step()
    engine.add("s1", list(request.all_token_ids) + prompt(2, 40), 17)
    for _ in range(3):
        engine.step()
    assert "s1" not in engine.admissions and tier._head is not None
    engine.add("late", prompt(80, 100), 4)
    request_ids = ("long", "s1", "late")
    for _ in range(3000):
        if all(engine.done(rid) for rid in request_ids):
            break
        engine.step()
        if tier._head is not None and tier._head[0] == "s1":
            assert "late" not in engine.admissions
    engine.quiesce()
    stats = tier.restore_stats
    assert stats.gave_up == 0 and stats.waited_behind > 0, stats
    # It resumed from RAM at the previous turn's exact end.
    assert stats.completed > 0, stats
    assert engine.admissions["s1"][0] >= len(request.all_token_ids) - 1
    assert tier._head is None and not tier._head_pins
    assert_nothing_lost(engine)


@pytest.mark.parametrize("hash_block_size", HASH_BLOCK_SIZES)
def test_the_prompt_floor_of_a_turn_that_starts_a_loop_outlives_its_run(
    hash_block_size: int,
):
    """The checkpoint graph lets a boundary state go once the request runs
    past it, and a GPU short of blocks then drops it. The state at the last
    boundary before the prompt end of a turn that starts a tool loop (a user
    prompt) is where the user prompt after the loop resumes when the states
    inside that block are gone: both tiers keep it from the turn's
    admission, so it is still there when the turn finishes."""
    engine = make_engine(
        False, hash_block_size, num_blocks=40, ram_state_spacing_blocks=0
    )
    roomy = roomy_engine(False, hash_block_size)
    tokens = prompt(31, 1000)
    for e in (engine, roomy):
        e.add("p0", tokens, 400)
        e.run("p0")
    floor = 1000 // BLOCK
    if floor not in _boundary_states(roomy, "p0"):
        pytest.skip("the prefill caches no state at the prompt floor")
    tier = manager(engine)
    request = engine.requests["p0"]
    key = request.block_hashes[floor * (BLOCK // hash_block_size) - 1]
    gpu = engine.scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block
    cpu = tier.cpu_block_pool.cached_block_hash_to_block
    for group_id in tier.mamba_gids:
        grouped = make_block_hash_with_group_id(key, group_id)
        assert gpu.get_one_block(grouped) or cpu.get_one_block(grouped), group_id
    assert_nothing_lost(engine)


def test_ram_never_copies_what_only_superseded_turns_resumed_at():
    """A tool loop's turns supersede each other: once the next turn finishes,
    the previous turn's exact end and prompt tail are where nothing resumes.
    The GPU drops their blocks without a copy, so RAM spends no block (and no
    eviction of a live session's) on them."""
    engine = make_engine(
        False,
        BLOCK // 4,
        num_blocks=36,
        num_cpu_blocks=48,
        ram_eviction="gdsf",
        ram_state_spacing_blocks=0,
    )
    tier = manager(engine)
    policy = tier.ram_policy
    assert policy is not None
    pool = engine.scheduler.kv_cache_manager.block_pool
    copied_disowned = []
    walk = tier._prepare_lazy_store_specs

    def record():
        specs = walk()
        for block_id in specs[0]:
            block = pool.blocks[block_id]
            extra = pool.cached_block_hashes_by_block.get(block_id, ())
            if policy.disowned((block.block_hash, *extra)):
                copied_disowned.append(block_id)
        return specs

    tier._prepare_lazy_store_specs = record  # type: ignore[method-assign]
    sessions = {s: prompt(40 + s, 400) for s in range(2)}
    for loop in range(3):
        for turn in range(3):
            for s in range(2):
                request_id = f"s{s}l{loop}t{turn}"
                engine.add(request_id, sessions[s], 20)
                engine.run(request_id)
                sessions[s] = list(engine.requests[request_id].all_token_ids) + prompt(
                    1000 + 100 * s + 10 * loop + turn, 40
                )
        for s in range(2):
            seed = 5000 + 100 * s + 10 * loop
            sessions[s] = sessions[s][:-43] + prompt(seed, 60)
    assert policy._disowned, "no turn was superseded"
    assert not copied_disowned
    assert_nothing_lost(engine)


@pytest.mark.parametrize("protect", [True, False])
def test_a_queued_turns_session_stays_on_the_gpu(protect: bool):
    """Session A finished before B, so LRU would move A to RAM first. A's
    next turn waits in the queue while a long prefill takes the blocks: the
    GPU moves B's pages to RAM instead (the queue's requests are the next to
    read their blocks), and A's turn resumes at its exact end with A's pages
    from the GPU. Only blocks RAM already copied may leave first (the spill
    guard's order): at most the exact-end point's own. Without that, A's
    pages come back from RAM too."""
    engine = make_engine(False, BLOCK, num_blocks=56, ram_state_spacing_blocks=0)
    tier = manager(engine)
    if not protect:
        tier.observe_queue = lambda waiting: None  # type: ignore[method-assign]
    engine.add("a0", prompt(1, 600), 17)
    engine.run("a0")
    engine.add("b0", prompt(2, 600), 17)
    engine.run("b0")
    a = engine.requests["a0"]
    engine.add("long", prompt(3, 1400), 4)
    engine.step()
    engine.add("a1", list(a.all_token_ids) + prompt(4, 30), 4)
    for _ in range(200):
        if engine.done("long") and engine.done("a1"):
            break
        engine.step()
    engine.quiesce()
    assert engine.admissions["a1"][0] >= len(a.all_token_ids) - 1
    gpu = engine.scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block
    b0 = engine.requests["b0"]
    b_pages = [
        make_block_hash_with_group_id(b0.block_hashes[i], tier.fa_gidx)
        for i in range(600 // BLOCK)
    ]
    b_moved = sum(gpu.get_one_block(key) is None for key in b_pages)
    stats = tier.restore_stats
    point = len(tier.mamba_gids) + 1
    if protect:
        assert stats.restored_blocks <= point and b_moved > 0, (stats, b_moved)
    else:
        assert stats.restored_blocks > point, stats
    assert_nothing_lost(engine)


@pytest.mark.parametrize("hash_block_size", HASH_BLOCK_SIZES)
@pytest.mark.parametrize("protect", [True, False])
def test_queued_turns_resume_whole_under_ram_pressure(
    protect: bool, hash_block_size: int
):
    """Four sessions fit both tiers (GDSF RAM). The two oldest sessions'
    next turns wait in the queue behind a long prefill that needs most of
    the GPU, and the GPU's evictions fill RAM: both tiers make room from the
    two idle sessions, so both queued turns resume at their exact ends,
    nothing queued is evicted and nothing leaves uncopied. Without that,
    the oldest session is the first to go. (The pools are sized so the idle
    sessions just cover what the prefill needs.)"""
    engine = make_engine(
        False,
        hash_block_size,
        num_blocks=56,
        num_cpu_blocks=52,
        ram_eviction="gdsf",
        ram_state_spacing_blocks=0,
    )
    tier = manager(engine)
    if not protect:
        tier.observe_queue = lambda waiting: None  # type: ignore[method-assign]
    for s in range(4):
        engine.add(f"s{s}", prompt(10 + s, 600), 17)
        engine.run(f"s{s}")
    engine.add("long", prompt(99, 1400), 4)
    engine.step()
    for s in (0, 1):
        tokens = list(engine.requests[f"s{s}"].all_token_ids) + prompt(20 + s, 30)
        engine.add(f"t{s}", tokens, 4)
    request_ids = ("long", "t0", "t1")
    for _ in range(400):
        if all(engine.done(rid) for rid in request_ids):
            break
        engine.step()
    engine.quiesce()
    deep = [
        engine.admissions[f"t{s}"][0]
        >= len(engine.requests[f"s{s}"].all_token_ids) - 1
        for s in (0, 1)
    ]
    if protect:
        assert all(deep), engine.admissions
        assert tier.ram_policy.stats.queued == 0
    else:
        assert not all(deep), engine.admissions
    assert_nothing_lost(engine)


def _queued_turn(engine: Engine) -> tuple[Request, list[Request]]:
    """A finished session's next turn, added to the queue but not stepped:
    the finished request and the waiting queue."""
    engine.add("a0", prompt(1, 600), 17)
    engine.run("a0")
    a0 = engine.requests["a0"]
    engine.add("a1", list(a0.all_token_ids) + prompt(2, 30), 4)
    return a0, list(engine.scheduler.waiting)


def test_a_queued_turn_looks_its_point_up_again_once_a_block_under_it_is_gone():
    """A queued turn's keys are found again once any of them left both
    tiers, not only its deepest: with a page gone under its point, the
    blocks above it resume nothing, and keeping them would keep them for
    nothing."""
    engine = make_engine(False, BLOCK, num_blocks=400)
    tier = manager(engine)
    a0, waiting = _queued_turn(engine)
    tier.observe_queue(waiting)
    page = make_block_hash_with_group_id(a0.block_hashes[3], tier.fa_gidx)
    assert page in tier._queued["a1"][1]
    pool = engine.scheduler.kv_cache_manager.block_pool
    block = pool.cached_block_hash_to_block.get_one_block(page)
    assert block is not None
    pool.evict_blocks({block.block_id})
    tier.observe_queue(waiting)
    assert page not in tier._queued["a1"][1]


def test_a_queued_turns_point_is_looked_up_again_every_requeue_steps():
    """A running request may compute a deeper point a queued turn can resume
    at: the turn's keys are found again every `_REQUEUE_STEPS` steps, also
    while it has some."""
    engine = make_engine(False, BLOCK, num_blocks=400)
    tier = manager(engine)
    _, waiting = _queued_turn(engine)
    lookups: list[str] = []
    resume_keys = tier._resume_keys

    def counted(request: Request) -> tuple:
        lookups.append(request.request_id)
        return resume_keys(request)

    tier._resume_keys = counted  # type: ignore[method-assign]
    for _ in range(2 * _REQUEUE_STEPS):
        tier.observe_queue(waiting)
    # When it joined, and `_REQUEUE_STEPS` steps later.
    assert lookups.count("a1") == 2
    assert tier._queued["a1"][1]


# A system prompt sessions start with: its last block boundary (448) is no
# prefill chunk's end (256 tokens per step).
SYSTEM = prompt(424242, 450)


def _sessions_with_a_system_prompt(engine: Engine, fillers: bool = False) -> list[int]:
    """Five sessions one after another, each the system prompt and its own
    first message, then a second turn; with `fillers`, two long prompts after
    each, which move the sessions out of a small GPU. Returns the tokens
    each session's first turn was admitted with."""
    for s in range(5):
        engine.add(f"s{s}t0", SYSTEM + prompt(600 + s, 150), 40)
        engine.run(f"s{s}t0")
        tokens = list(engine.requests[f"s{s}t0"].all_token_ids) + prompt(700 + s, 30)
        engine.add(f"s{s}t1", tokens, 20)
        engine.run(f"s{s}t1")
        for f in range(2 if fillers else 0):
            engine.add(f"f{s}{f}", prompt(800 + 10 * s + f, 1300), 4)
            engine.run(f"f{s}{f}")
    return [engine.admissions[f"s{s}t0"][0] for s in range(5)]


@pytest.mark.parametrize("hash_block_size", HASH_BLOCK_SIZES)
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_sessions_that_share_a_prefix_resume_at_its_last_block_boundary(
    async_scheduling: bool, hash_block_size: int
):
    """Sessions start one after another with the same system prompt. The
    second finds its pages with no state under them (the first's were
    released when it finished) and caches the state at their last block
    boundary, the junction: both tiers keep it, so from the third on
    sessions resume there, as on a GPU that never evicts."""
    engine = make_engine(
        async_scheduling, hash_block_size, num_blocks=400, ram_state_spacing_blocks=0
    )
    resumed = _sessions_with_a_system_prompt(engine)
    expected = _sessions_with_a_system_prompt(
        roomy_engine(async_scheduling, hash_block_size)
    )
    assert resumed[2:] == expected[2:], (resumed, expected)
    assert min(expected[2:]) >= 7 * BLOCK
    assert len(manager(engine)._junctions) == 1
    assert_nothing_lost(engine)


@pytest.mark.parametrize("ram_eviction", ["lru", "gdsf"])
def test_a_shared_prefix_state_comes_back_from_ram(ram_eviction: str):
    """As above, with a GPU that two long prompts after each session empty of
    it: the shared pages are in RAM alone when the next session arrives, and
    so is the junction's state from the third session on. The junction is
    found across both tiers, and the sessions resume at it from RAM."""
    engine = make_engine(
        False,
        BLOCK,
        num_blocks=40,
        ram_eviction=ram_eviction,
        ram_state_spacing_blocks=0,
    )
    resumed = _sessions_with_a_system_prompt(engine, fillers=True)
    expected = _sessions_with_a_system_prompt(roomy_engine(False, BLOCK), fillers=True)
    assert resumed[2:] == expected[2:], (resumed, expected)
    assert manager(engine).restore_stats.completed >= 3
    assert_nothing_lost(engine)


def test_a_compaction_resumes_at_the_head_once_it_is_known():
    """A session runs a few turns, so only its latest points are kept, then
    restarts as its head (the system prompt) and a summary, twice. The first
    restart finds the head's pages without a state and caches it; the second
    restart, and a new session after it, resume there."""
    engines = (
        make_engine(False, BLOCK, num_blocks=400, ram_state_spacing_blocks=0),
        roomy_engine(False, BLOCK),
    )
    for e in engines:
        tokens = SYSTEM + prompt(600, 150)
        for turn in range(4):
            e.add(f"t{turn}", tokens, 40)
            e.run(f"t{turn}")
            tokens = list(e.requests[f"t{turn}"].all_token_ids) + prompt(610 + turn, 30)
        for c in range(2):
            e.add(f"c{c}", SYSTEM + prompt(620 + c, 120), 20)
            e.run(f"c{c}")
        e.add("n", SYSTEM + prompt(630, 150), 20)
        e.run("n")
    engine, roomy = engines
    for rid in ("c1", "n"):
        assert engine.admissions[rid][0] == roomy.admissions[rid][0] >= 7 * BLOCK
    assert_nothing_lost(engine)


def test_a_second_edit_resumes_where_the_first_parted():
    """A message 1,200 tokens into a session is edited twice. The first edit
    resumes at the last state kept before it and caches the state at the
    last block boundary the two share; the second resumes there, as on a GPU
    that never evicts."""
    engine = make_engine(False, BLOCK, num_blocks=400)
    roomy = roomy_engine(False, BLOCK)
    tokens = prompt(13, 1500)
    for e in (engine, roomy):
        e.add("d0", tokens, 40)
        e.run("d0")
        for edit in (1, 2):
            e.add(f"d{edit}", tokens[:1200] + prompt(13 + edit, 60), 17)
            e.run(f"d{edit}")
    assert engine.admissions["d2"][0] == roomy.admissions["d2"][0] == 18 * BLOCK
    assert_nothing_lost(engine)


def test_a_sessions_own_re_read_keeps_at_most_one_state_more():
    """A session whose states both tiers lost re-reads its own pages: its
    turn marks the pages' last block boundary as a junction, one state more
    than its own points, and its next turn resumes at its exact end without
    marking another."""
    engine = make_engine(False, BLOCK, num_blocks=400, ram_state_spacing_blocks=0)
    tier = manager(engine)
    engine.add("d0", prompt(13, 1500), 40)
    engine.run("d0")
    for pool in (engine.scheduler.kv_cache_manager.block_pool, tier.cpu_block_pool):
        pool.evict_blocks(
            {
                b.block_id
                for b in pool.blocks
                if b.block_hash is not None
                and get_group_id(b.block_hash) in tier.mamba_gids
            }
        )
    d1 = list(engine.requests["d0"].all_token_ids) + prompt(16, 100)
    engine.add("d1", d1, 17)
    engine.run("d1")
    assert engine.admissions["d1"][0] == 0
    assert len(tier._junctions) == 1
    # d1 extends d0's end, a tool loop's next turn: its own points are the
    # floors before the loop's start (d0's prompt, 23) and its own prompt
    # end (25). The one more is the pages' last boundary (24).
    assert set(_cached_states(engine, "d1")) - {23, 25} <= {24}
    engine.add("d2", list(engine.requests["d1"].all_token_ids) + prompt(17, 40), 17)
    engine.run("d2")
    assert engine.admissions["d2"][0] >= len(engine.requests["d1"].all_token_ids) - 1
    assert len(tier._junctions) == 1
    assert_nothing_lost(engine)
