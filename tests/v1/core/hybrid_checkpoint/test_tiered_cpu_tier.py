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
from vllm.v1.core.kv_cache_utils import init_none_hash, make_block_hash_with_group_id
from vllm.v1.simple_kv_offload.tiered import TieredCPUOffloadScheduler

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
