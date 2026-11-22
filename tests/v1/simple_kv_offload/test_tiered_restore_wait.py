# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A restored session stays pinned for admission for a share of the prefill
it saves, within bounds. How restores wait for room and pass each other is
measured on a real engine (tests/v1/hybrid_gpu/test_eager_vs_lazy.py)."""

import pytest

from vllm.v1.core.hybrid_checkpoint.cost import PrefillCostModel
from vllm.v1.simple_kv_offload.tiered import (
    _MAX_PIN_TIMEOUT_S,
    _PIN_TIMEOUT_S,
    pin_timeout_s,
)

# Production's fitted coefficients: ~136 s to prefill 120k tokens.
COST = PrefillCostModel(7.4893e-04, 6.4382e-04)


def test_the_pin_timeout_grows_with_the_prefill_a_restore_saves():
    small, session = pin_timeout_s(COST, 0, 4_000), pin_timeout_s(COST, 0, 120_000)
    assert small == _PIN_TIMEOUT_S
    assert session == pytest.approx(0.5 * COST.seconds(0, 120_000))
    assert _PIN_TIMEOUT_S < session < _MAX_PIN_TIMEOUT_S
    # Only the part the GPU lacks counts.
    assert pin_timeout_s(COST, 110_000, 120_000) == _PIN_TIMEOUT_S
    assert pin_timeout_s(COST, 0, 250_000) == _MAX_PIN_TIMEOUT_S
    assert pin_timeout_s(None, 0, 250_000) == _PIN_TIMEOUT_S
