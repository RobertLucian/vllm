# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assertions shared by the tests."""

from __future__ import annotations

import json
import os
from typing import Any

from .workloads import Result


def assert_outputs_close(
    reference: Result, cached: Result, name: str = "cached"
) -> None:
    """Greedy outputs agree, or part at a near tie: where the tokens first
    differ, each run's token is among the other's top logprobs (the rule of
    tests/models/utils.py `check_logprobs_close`)."""
    assert reference.logprobs is not None and cached.logprobs is not None
    for i, (ref, got) in enumerate(zip(reference.output, cached.output)):
        if ref == got:
            continue
        assert ref in cached.logprobs[i] and got in reference.logprobs[i], (
            f"the {name} run parts from the cold run at output token {i}: "
            f"{got} vs {ref}; top logprobs {cached.logprobs[i]} vs "
            f"{reference.logprobs[i]}"
        )
        return


def delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Counters that changed, by how much."""
    return {
        k: after[k] - before.get(k, 0) for k in after if after[k] != before.get(k, 0)
    }


def save_result(test: str, arm: str, summary: dict[str, Any]) -> None:
    """With `HYBRID_GPU_RESULTS` set, append one JSON line per measured arm
    there as it finishes: a run cut off by its time limit keeps what it
    measured (the JUnit report is written only at the end)."""
    path = os.environ.get("HYBRID_GPU_RESULTS")
    if not path:
        return
    with open(path, "a") as f:
        f.write(json.dumps({"test": test, "arm": arm, **summary}, default=str) + "\n")
