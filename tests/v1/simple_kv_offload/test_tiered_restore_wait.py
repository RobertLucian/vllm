# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How long a RAM -> VRAM restore may wait for room, and stay pinned for
admission: a share of the prefill it saves, within bounds. A restore that
saves a long session's prefill waits longer than one that saves a little;
without a cost model every restore gets the minimum."""

from types import SimpleNamespace

import pytest

from vllm.v1.core.hybrid_checkpoint.cost import PrefillCostModel
from vllm.v1.simple_kv_offload import tiered
from vllm.v1.simple_kv_offload.tiered import (
    _MAX_RESTORE_WAIT_S,
    _MIN_RESTORE_WAIT_S,
    RestoreStats,
    TieredCPUOffloadScheduler,
    _Plan,
)

# Production's fitted coefficients: ~136 s to prefill 120k tokens.
COST = PrefillCostModel(7.4893e-04, 6.4382e-04)


def wait_s(cost, gpu_only, num_tokens):
    plan = _Plan(num_tokens, [], gpu_only_tokens=gpu_only)
    return TieredCPUOffloadScheduler._restore_wait_s(SimpleNamespace(_cost=cost), plan)


def test_the_wait_grows_with_the_prefill_a_restore_saves():
    small, session = wait_s(COST, 0, 4_000), wait_s(COST, 0, 120_000)
    assert small == _MIN_RESTORE_WAIT_S
    assert session == pytest.approx(0.5 * COST.seconds(0, 120_000))
    assert _MIN_RESTORE_WAIT_S < session < _MAX_RESTORE_WAIT_S
    # Only the part the GPU lacks counts.
    assert wait_s(COST, 110_000, 120_000) == _MIN_RESTORE_WAIT_S


def test_the_wait_is_bounded():
    assert wait_s(COST, 0, 250_000) == _MAX_RESTORE_WAIT_S
    assert wait_s(None, 0, 250_000) == _MIN_RESTORE_WAIT_S


def test_a_request_gives_up_only_after_its_own_wait(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(tiered.time, "monotonic", lambda: now[0])
    manager = SimpleNamespace(_waiting_since={}, restore_stats=RestoreStats())

    def wait(request_id, budget):
        return TieredCPUOffloadScheduler._wait(manager, request_id, budget)

    assert wait("long", 60.0) == (None, False)
    assert wait("short", 10.0) == (None, False)
    now[0] += 30.0
    # Past the short one's budget, inside the long one's.
    assert wait("short", 10.0) == (0, False)
    assert wait("long", 60.0) == (None, False)
    now[0] += 31.0
    assert wait("long", 60.0) == (0, False)
    assert manager.restore_stats.gave_up == 2
    assert manager._waiting_since == {}
