# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba frontier checkpoints under speculative decoding.

The real scheduler with three draft tokens per step from an n-gram drafter,
and a stand-in for the device (`spec_device`) that keeps one state column per
draft position and checks every state read and every cached state against its
token position. Sessions are multi-turn: a turn's prompt is the previous
turn's prompt and output plus new tokens.
"""

import random

import pytest

from vllm.config import KVTransferConfig, SpeculativeConfig
from vllm.utils.hashing import sha256
from vllm.v1.core.hybrid_checkpoint.speculative import MAX_MIRRORED_STOP_TOKEN_IDS
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.request import RequestStatus
from vllm.v1.worker.gpu.sample.logit_bias import MAX_NUM_STOP_TOKEN_IDS

from .spec_device import STOP, Device, Engine, Fault
from .utils import make_hybrid_scheduler, mock_connector, prompt, simple_cpu_connector

pytestmark = pytest.mark.cpu_test

BLOCK = 64
DRAFTS = 3
COMPRESSION_RATIO = 4
# The ring holds two compressed groups: a step stores its drafts' rows before
# acceptance is known.
RING = 8
# GreedyDual-Size prefill cost (seconds per token, per token per 100k context).
COST = (1.862e-5, 3.33e-6)


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def speculative_config(truncate: bool = True) -> SpeculativeConfig:
    # A drafter without the EAGLE block drop, which asynchronous scheduling
    # supports.
    return SpeculativeConfig(
        method="ngram_gpu",
        num_speculative_tokens=DRAFTS,
        truncate_acceptance_at_stop=truncate,
    )


def default_accept(request_id: str, start: int) -> int:
    return (sum(map(ord, request_id)) * 7 + start * 13) % (DRAFTS + 1)


def make_engine(
    *,
    async_scheduling: bool = True,
    num_blocks: int = 400,
    truncate: bool = True,
    accept=default_accept,
    fault: Fault | None = None,
    ring: int = RING,
    kv_transfer_config: KVTransferConfig | None = None,
    eviction: str = "lru",
    max_num_batched_tokens: int = 1024,
) -> Engine:
    scheduler = make_hybrid_scheduler(
        block_size=BLOCK,
        num_blocks=num_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
        ring=ring,
        ring_compression_ratio=COMPRESSION_RATIO,
        speculative_config=speculative_config(),
        num_speculative_blocks=DRAFTS,
        async_scheduling=async_scheduling,
        mamba_frontier_checkpoint=True,
        kv_transfer_config=kv_transfer_config,
        prefix_cache_eviction=eviction,
        prefill_cost_coefficients=COST if eviction == "gds" else None,
    )
    device = Device(
        scheduler, num_drafts=DRAFTS, truncate=truncate, accept=accept, fault=fault
    )
    return Engine(scheduler, device, async_scheduling)


def session(
    engine: Engine,
    sid: int = 1,
    turns: int = 3,
    first: int = 900,
    out: tuple[int, ...] = (30, 41, 77),
    extra: tuple[int, ...] = (70, 33, 51),
    eos: dict[int, int] | None = None,
    stop_token_ids: list[int] | None = None,
) -> list[int]:
    """Run a session; returns each turn's length. `eos`: turn -> the output
    position at which the model samples EOS, or the first of
    `stop_token_ids` when given."""
    tokens = prompt(sid, first)
    lengths = []
    for turn in range(turns):
        request_id = f"s{sid}t{turn}"
        eos_at = None
        if eos is not None and turn in eos:
            eos_at = len(tokens) + eos[turn]
        engine.add(
            request_id,
            tokens,
            out[turn % len(out)],
            eos_at=eos_at,
            stop_token_ids=stop_token_ids if eos_at is not None else None,
        )
        engine.run(request_id)
        all_tokens = list(engine.requests[request_id].all_token_ids)
        lengths.append(len(all_tokens))
        tokens = all_tokens + prompt(100 * sid + turn, extra[turn % len(extra)])
    return lengths


def assert_clean(engine: Engine) -> None:
    device = engine.device
    assert not device.violations, device.violations[:5]
    assert device.counts["reads"] > 0 and device.counts["cached_checks"] > 0


def checkpoints(engine: Engine):
    manager = engine.scheduler.mamba_checkpoints
    assert manager is not None and manager.speculative is not None
    return manager


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_session_resumes_at_its_frontiers(async_scheduling: bool):
    engine = make_engine(async_scheduling=async_scheduling)
    lengths = session(engine)
    assert_clean(engine)
    manager = checkpoints(engine)
    assert manager.frontiers.num_published == 3, manager.speculative.declined
    assert manager.frontiers.num_hits == 2
    for turn in (1, 2):
        # With a step in flight at the finish its first input token, the
        # request's last token, is in the state; otherwise it is not.
        (admitted,) = engine.admissions[f"s1t{turn}"]
        assert admitted in (lengths[turn - 1], lengths[turn - 1] - 1)
    if not async_scheduling:
        assert engine.admissions["s1t1"] == [lengths[0] - 1]
        # The state is in an accepted draft column, folded into the first.
        assert engine.device.counts["gathers"] > 0


@pytest.mark.parametrize("stop", ["eos", "stop_token_id"])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_stop_inside_accepted_drafts(async_scheduling: bool, stop: str):
    """Every draft is accepted, so the stop (EOS or a stop token id) lands
    inside an accepted run; the truncated acceptance keeps the published
    state exact."""
    engine = make_engine(
        async_scheduling=async_scheduling,
        accept=lambda request_id, start: DRAFTS,
    )
    stop_token_ids = [STOP] if stop == "stop_token_id" else None
    session(
        engine, out=(500, 400, 400), eos={0: 22, 1: 9}, stop_token_ids=stop_token_ids
    )
    assert_clean(engine)
    first = engine.requests["s1t0"]
    assert first.status == RequestStatus.FINISHED_STOPPED
    assert len(first.output_token_ids) == 23
    manager = checkpoints(engine)
    assert manager.frontiers.num_published == 3, manager.speculative.declined


def test_more_stop_token_ids_than_the_device_holds():
    """With more stop token ids than the device mirrors, a stop id is not cut
    on the device: inside the accepted drafts, or as a step's last token with
    the next step in flight, which then runs past it. Either way the frontier
    is declined, never published wrong."""
    engine = make_engine(accept=lambda request_id, start: DRAFTS)
    stop_token_ids = list(range(STOP, STOP + MAX_MIRRORED_STOP_TOKEN_IDS + 1))
    # Eight stop positions cover two whole verify steps of 1 + DRAFTS tokens.
    # At 956 (output 56) the stop ends a step, and the step in flight ends at
    # the block boundary 960, where the device folds its last draft column
    # into the column the request finished with.
    assert (900 + 56 + 1 + DRAFTS) % BLOCK == 0
    for sid, at in enumerate(range(53, 61), start=1):
        session(
            engine,
            sid=sid,
            turns=1,
            out=(500,),
            eos={0: at},
            stop_token_ids=stop_token_ids,
        )
        assert engine.requests[f"s{sid}t0"].status == RequestStatus.FINISHED_STOPPED
    assert_clean(engine)
    declined = checkpoints(engine).speculative.declined
    assert declined["stop_inside_accepted"] > 0
    assert declined["not_mirrored_stop"] > 0


def test_the_mirrored_stop_token_id_limit_matches_the_gpu_sampler():
    assert MAX_MIRRORED_STOP_TOKEN_IDS == MAX_NUM_STOP_TOKEN_IDS


def test_without_truncation_a_stop_inside_accepted_drafts_publishes_nothing():
    """A device that runs past the stop (startup refuses this configuration;
    here only the device misbehaves): the frontier is declined, never
    published wrong."""
    engine = make_engine(truncate=False, accept=lambda request_id, start: DRAFTS)
    session(engine, out=(500, 400, 400), eos={0: 22, 1: 9})
    assert_clean(engine)
    assert checkpoints(engine).speculative.declined["stop_inside_accepted"] >= 1


def test_decode_boundary_state_at_a_crossed_boundary():
    """The output crosses several block boundaries; the state the last verify
    step left at the last of them is published as a decode-boundary state.
    Without a step in flight, align mode frees the block that holds it before
    it could cache it, so only the snapshot keeps it."""
    engine = make_engine(async_scheduling=False)
    tokens = prompt(3, 3 * BLOCK - 40)
    engine.add("leaf", tokens, 300)
    engine.run("leaf")
    assert_clean(engine)
    num_tokens = len(engine.requests["leaf"].all_token_ids)
    boundary = (num_tokens - 1) // BLOCK * BLOCK
    units = checkpoints(engine).graph.units.values()
    assert any(u.num_tokens == boundary and not u.frontier for u in units), [
        u.num_tokens for u in units
    ]


def test_concurrent_sessions_under_pool_pressure():
    rng = random.Random(11)
    engine = make_engine(
        num_blocks=180,
        accept=lambda request_id, start: hash((request_id, start)) % (DRAFTS + 1),
    )
    tokens = {s: prompt(s, rng.randrange(200, 1400)) for s in range(10)}
    for turn in range(3):
        request_ids = [f"c{s}t{turn}" for s in range(10)]
        for s, request_id in enumerate(request_ids):
            engine.add(request_id, tokens[s], rng.choice((17, 90, 200)))
        engine.run(*request_ids)
        for s, request_id in enumerate(request_ids):
            tokens[s] = list(engine.requests[request_id].all_token_ids) + prompt(
                1000 + 10 * s + turn, rng.randrange(10, 90)
            )
    assert_clean(engine)
    manager = checkpoints(engine)
    assert manager.frontiers.num_published > 0 and manager.frontiers.num_hits > 0


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_sessions_resume_through_the_cpu_tier(async_scheduling: bool, lazy: bool):
    """With SimpleCPU offloading, sessions overflow a small GPU pool into the
    CPU tier and come back from it. Every state the connector stores, every
    state a loaded request reads and every cached state holds its position."""
    rng = random.Random(23)
    engine = make_engine(
        async_scheduling=async_scheduling,
        num_blocks=180,
        ring=0,
        kv_transfer_config=simple_cpu_connector(2000, lazy=lazy),
    )
    tokens = {s: prompt(s, rng.randrange(200, 1400)) for s in range(10)}
    for turn in range(3):
        request_ids = [f"c{s}t{turn}" for s in range(10)]
        for s, request_id in enumerate(request_ids):
            engine.add(request_id, tokens[s], rng.choice((17, 90, 200)))
        engine.run(*request_ids)
        for s, request_id in enumerate(request_ids):
            tokens[s] = list(engine.requests[request_id].all_token_ids) + prompt(
                1000 + 10 * s + turn, rng.randrange(10, 90)
            )
    assert_clean(engine)
    counts = engine.device.counts
    assert counts["store_checks"] > 0 and counts["loads"] > 0, counts
    assert checkpoints(engine).frontiers.num_hits > 0


@pytest.mark.parametrize("eviction", ["lru", "gds"])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_the_lazy_cpu_tier_evicts_no_block_it_could_restore(
    async_scheduling: bool, eviction: str, monkeypatch: pytest.MonkeyPatch
):
    """Lazy SimpleCPU offloading extends the GPU pool: sessions rotate through
    a pool that holds about three of them. No cached block a CPU lookup could
    restore leaves the GPU before the CPU pool holds it; a resume that needs
    more blocks than are safe to evict waits for the stores. Every resume
    goes at least as deep as with an eager CPU tier, which mirrors every
    block it computes."""

    def rotation_engine(lazy: bool) -> Engine:
        return make_engine(
            async_scheduling=async_scheduling,
            num_blocks=90,
            ring=0,
            kv_transfer_config=simple_cpu_connector(4000, lazy=lazy),
            eviction=eviction,
            max_num_batched_tokens=256,
        )

    def rotate(engine: Engine) -> list[int]:
        """Six sessions, three turns each, one session after another; returns
        the computed tokens each later turn was admitted with."""
        rng = random.Random(5)
        tokens = {s: prompt(s, rng.randrange(1000, 1400)) for s in range(6)}
        resumed = []
        for turn in range(3):
            for s in range(6):
                request_id = f"l{s}t{turn}"
                engine.add(request_id, tokens[s], rng.choice((17, 90)))
                engine.run(request_id)
                if turn:
                    resumed.append(engine.admissions[request_id][-1])
                tokens[s] = list(engine.requests[request_id].all_token_ids) + prompt(
                    2000 + 10 * s + turn, rng.randrange(10, 90)
                )
        return resumed

    engine = rotation_engine(lazy=True)
    pool = engine.scheduler.kv_cache_manager.block_pool
    guard = pool.spill_guard
    assert guard is not None
    # A short lead: a load from the CPU tier needs more blocks than the walk
    # keeps free or stored ahead of it.
    guard._target_free = 4
    lost: list[int] = []
    evict = pool._maybe_evict_cached_block

    def checked_evict(block) -> bool:
        # A block a request holds can lose a key (an unwritten state), not
        # leave the GPU.
        if guard.holding and guard.needs_spill(block):
            lost.append(block.block_id)
        return evict(block)

    monkeypatch.setattr(pool, "_maybe_evict_cached_block", checked_evict)
    resumed = rotate(engine)
    assert_clean(engine)
    assert not lost
    stats = guard.lazy_spill_stats
    assert stats.unspilled_evicted_blocks == 0, stats
    assert stats.stored_blocks > 0 and stats.held_allocations > 0, stats
    assert engine.device.counts["loads"] > 0
    mirrored = rotate(rotation_engine(lazy=False))
    assert all(lazy >= eager for lazy, eager in zip(resumed, mirrored)), (
        resumed,
        mirrored,
    )


@pytest.mark.parametrize("fault", ["skip_gather", "no_conv_shift"])
def test_device_faults_are_detected(fault: Fault):
    """A gather skipped or applied without the window shift leaves a published
    state wrong, which the device's checks catch."""
    engine = make_engine(async_scheduling=False, fault=fault)
    session(engine)
    counts = engine.device.counts
    assert counts["gathers"] > 0
    assert counts["violation_cached"] + counts["violation_read"] > 0


def test_abort_with_a_step_in_flight_publishes_no_frontier():
    engine = make_engine()
    engine.add("aborted", prompt(5, 600), 2000)
    request = engine.requests["aborted"]
    while not (request.num_output_tokens > 50 and request.num_in_flight_tokens):
        engine.step()
    engine.scheduler.finish_requests("aborted", RequestStatus.FINISHED_ABORTED)
    engine.quiesce()
    assert_clean(engine)
    assert checkpoints(engine).frontiers.num_published == 0


def test_a_reset_with_a_gather_pending_drops_its_frontier():
    """A request finishes with its state in a draft column, and the prefix
    cache is reset before the step that would gather it into the first."""
    engine = make_engine(async_scheduling=False, accept=lambda rid, start: DRAFTS)
    engine.add("r", prompt(1, 900), 41)
    while not engine.done("r"):
        engine.step()
    manager = checkpoints(engine)
    assert manager.frontiers.pending_gathers
    assert engine.scheduler.reset_prefix_cache(reset_running_requests=True)
    assert not manager.frontiers.pending_gathers and not manager.frontiers.entries
    pool = engine.scheduler.kv_cache_manager.block_pool
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    # The engine goes on: the next request's states are all exact.
    engine.add("next", prompt(1, 900), 41)
    engine.run("next")
    assert_clean(engine)


def test_an_aborted_request_that_was_looked_up_leaves_no_lookup():
    engine = make_engine()
    session(engine, turns=1)
    request = engine.add("waiting", prompt(1, 900) + prompt(9, 40), 30)
    # The scheduler looked it up, then could not admit it.
    engine.scheduler.kv_cache_manager.get_computed_blocks(request)
    manager = checkpoints(engine)
    assert "waiting" in manager.frontiers._lookups
    engine.scheduler.finish_requests("waiting", RequestStatus.FINISHED_ABORTED)
    assert not manager.frontiers._lookups


def test_resume_inside_an_open_group_catches_up_to_the_group_not_the_ring():
    """Under speculative decoding the ring holds two groups; a request resumed
    inside an open group takes one token per step only until the group
    closes."""
    engine = make_engine(async_scheduling=False)
    lengths = session(engine, turns=2)
    manager = checkpoints(engine)
    assert manager.catchup_pool_size == COMPRESSION_RATIO
    (admitted,) = engine.admissions["s1t1"]
    assert admitted == lengths[0] - 1
    assert_clean(engine)


@pytest.mark.parametrize(
    ("kwargs", "needs"),
    [
        (
            dict(kv_transfer_config=mock_connector(BLOCK)),
            "needs a KV connector that supports frontier checkpoints",
        ),
        (
            dict(resume_catchup="prefill"),
            "indexer rings that only the decode path reads",
        ),
    ],
)
def test_unsupported_speculative_configurations_are_refused(kwargs, needs):
    with pytest.raises(ValueError, match=needs):
        make_hybrid_scheduler(
            block_size=BLOCK,
            ring=RING,
            speculative_config=speculative_config(),
            num_speculative_blocks=DRAFTS,
            mamba_frontier_checkpoint=True,
            **kwargs,
        )
