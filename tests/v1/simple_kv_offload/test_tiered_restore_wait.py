# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How a RAM -> VRAM restore waits for room. It waits as long as blocks take
to free up (a backstop bounds it), and the first such request in the queue
holds the requests behind it back, as a load that waits for blocks does;
earlier ones (a chat under the priority policy) pass. A restored session
stays pinned for admission for a share of the prefill it saves, within
bounds."""

from functools import partial
from types import SimpleNamespace

import pytest

from vllm.v1.core.hybrid_checkpoint.cost import PrefillCostModel
from vllm.v1.simple_kv_offload import tiered
from vllm.v1.simple_kv_offload.tiered import (
    _MAX_PIN_TIMEOUT_S,
    _MAX_RESTORE_WAIT_S,
    _PIN_TIMEOUT_S,
    RestoreStats,
    TieredCPUOffloadScheduler,
    _Plan,
)

# Production's fitted coefficients: ~136 s to prefill 120k tokens.
COST = PrefillCostModel(7.4893e-04, 6.4382e-04)


def pin_timeout_s(cost, gpu_only, num_tokens):
    plan = _Plan(num_tokens, [], gpu_only_tokens=gpu_only)
    return TieredCPUOffloadScheduler._restore_wait_s(SimpleNamespace(_cost=cost), plan)


def test_the_pin_timeout_grows_with_the_prefill_a_restore_saves():
    small, session = pin_timeout_s(COST, 0, 4_000), pin_timeout_s(COST, 0, 120_000)
    assert small == _PIN_TIMEOUT_S
    assert session == pytest.approx(0.5 * COST.seconds(0, 120_000))
    assert _PIN_TIMEOUT_S < session < _MAX_PIN_TIMEOUT_S
    # Only the part the GPU lacks counts.
    assert pin_timeout_s(COST, 110_000, 120_000) == _PIN_TIMEOUT_S
    assert pin_timeout_s(COST, 0, 250_000) == _MAX_PIN_TIMEOUT_S
    assert pin_timeout_s(None, 0, 250_000) == _PIN_TIMEOUT_S


def make_manager():
    pool = SimpleNamespace(pinned=[])
    pool.get_num_free_blocks = lambda: 100 - len(pool.pinned)
    pool.touch = lambda blocks: pool.pinned.extend(blocks)
    pool.free_blocks = lambda blocks: [pool.pinned.remove(b) for b in blocks]
    manager = SimpleNamespace(
        _waiting_since={},
        restore_stats=RestoreStats(),
        _head=None,
        _head_pins=[],
        cpu_block_pool=pool,
        _priority_policy=True,
    )
    for name in ("_queue_position", "_behind_head", "_stop_waiting", "_unpin_head"):
        setattr(
            manager, name, partial(getattr(TieredCPUOffloadScheduler, name), manager)
        )
    manager.wait = partial(TieredCPUOffloadScheduler._wait, manager)
    return manager


def request(request_id, priority, arrival_time):
    return SimpleNamespace(
        request_id=request_id, priority=priority, arrival_time=arrival_time
    )


def test_a_restore_waits_for_room_until_the_backstop(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(tiered.time, "monotonic", lambda: now[0])
    manager = make_manager()
    agent = request("agent", 10, 1.0)
    assert manager.wait(agent, ["b1", "b2"]) == (None, False)
    # Far past the old 10-120 s budgets: it still waits, its RAM blocks pinned.
    now[0] += 300.0
    assert manager.wait(agent, ["b1", "b2"]) == (None, False)
    assert manager._head[0] == "agent"
    assert sorted(manager.cpu_block_pool.pinned) == ["b1", "b2"]
    now[0] += _MAX_RESTORE_WAIT_S
    assert manager.wait(agent, ["b1", "b2"]) == (0, False)
    assert manager.restore_stats.gave_up == 1
    assert manager._head is None and manager._waiting_since == {}
    assert manager.cpu_block_pool.pinned == []


def test_the_first_waiting_request_in_the_queue_holds_the_others_back(monkeypatch):
    monkeypatch.setattr(tiered.time, "monotonic", lambda: 100.0)
    manager = make_manager()
    first, later = request("first", 10, 1.0), request("later", 10, 2.0)
    chat = request("chat", 0, 3.0)
    assert manager.wait(first, ["a"]) == (None, False)
    assert manager._behind_head(later)
    assert not manager._behind_head(first)
    # Under the priority policy a chat comes first in the queue.
    assert not manager._behind_head(chat)
    # A chat that waits for room too comes first: the agent waits behind it,
    # its RAM blocks no longer pinned.
    assert manager.wait(chat, ["c"]) == (None, False)
    assert manager._head[0] == "chat" and manager._behind_head(first)
    assert manager.cpu_block_pool.pinned == ["c"]
    manager._stop_waiting("chat")
    assert manager._head is None and manager.cpu_block_pool.pinned == []
    assert not manager._behind_head(later)
    # The children of a request with n > 1 arrive together: ordered by id.
    assert manager.wait(request("req-0", 10, 5.0), ["d"]) == (None, False)
    assert manager._behind_head(request("req-1", 10, 5.0))


def test_a_waiting_restore_pins_its_ram_blocks_only_if_ram_takes_the_copies(
    monkeypatch,
):
    """Room on the GPU may take copies into RAM first: a restore that would
    pin so much of RAM that those copies no longer fit waits unpinned."""
    monkeypatch.setattr(tiered.time, "monotonic", lambda: 100.0)
    manager = make_manager()
    manager.cpu_block_pool.get_num_free_blocks = lambda: 60
    big = [f"b{i}" for i in range(40)]
    assert manager.wait(request("big", 10, 1.0), big) == (None, False)
    assert manager._head[0] == "big" and manager.cpu_block_pool.pinned == []
