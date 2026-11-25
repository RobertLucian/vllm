# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The bounded resume search decides as the whole walk does, and the
connector's metrics and logs hold up on a busy run.

A lookup searches down from the deepest point a request can resume at only
as far as the restore gain it checks (`TieredCPUOffloadScheduler._search`):
no point below that changes the decision, so it stops there, also scanning no
hash unit below it. With `verify_search` every bounded search also runs the
whole walk; on window H's workload, for 15 s, they must agree on every
lookup, and the bounded one must visit fewer points and scan fewer units.

The same run has vLLM's stat loggers on, so Prometheus observes every step's
connector stats through the production path (`KVConnectorProm`), which
raises on a key or label it does not know; and it logs every restore and a
summary line every few seconds, so their formatting runs too.
"""

from __future__ import annotations

import logging

import pytest
from prometheus_client import REGISTRY

from vllm.v1.simple_kv_offload import tiered

from .test_head_of_line import _A_AGENTS, _A_GPU, _A_RAM, _report, _run, _tiered

_SECONDS = 15.0
_SUMMARY_S = 5.0


def _prometheus_total(name: str) -> float:
    """A counter's value summed over its labels (0 if never set)."""
    total = 0.0
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name == name:
                total += sample.value
    return total


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.mark.parametrize("tp", [1, 2], ids=lambda v: f"tp{v}")
def test_bounded_search_agrees(engines, tp, record_property):
    spec = _tiered(
        tp,
        _A_GPU,
        _A_RAM,
        admission="pack",
        verify_search=True,
        log_restores=True,
        summary_log_s=_SUMMARY_S,
        log_stats=True,
    )
    # After the engine starts: its logging setup clears the handlers of
    # vLLM's loggers (`dictConfig`), so a handler added before is gone.
    engines.get(spec)
    handler = _Lines()
    tiered.logger.addHandler(handler)
    try:
        summary = _run(engines, spec, _A_AGENTS, max_tokens=1, seconds=_SECONDS)
    finally:
        tiered.logger.removeHandler(handler)
    tiered_lines = handler.lines
    lookups = _prometheus_total("vllm:tiered_kv_lookups_total")
    seconds = _prometheus_total("vllm:tiered_kv_scheduler_seconds_total")
    admissions = _prometheus_total("vllm:tiered_kv_admission_total")
    searches = _prometheus_total("vllm:tiered_kv_search_total")
    summaries = [line for line in tiered_lines if line.startswith("Tiered KV, last")]
    restore_lines = [line for line in tiered_lines if "Tiered KV restore" in line]
    summary.update(
        prometheus=dict(
            lookups=lookups, seconds=seconds, admission=admissions, search=searches
        ),
        summary_lines=len(summaries),
        restore_lines=len(restore_lines),
    )
    _report(f"bounded-search tp{tp}", {"pack_verified": summary}, record_property)
    for line in summaries:
        print(line)

    counters = summary["counters"]
    assert summary["short_resumes"] == 0, summary
    assert counters.get("spill.unspilled_evicted_blocks", 0) == 0, summary
    assert counters.get("restore.completed", 0) > 0, summary
    # The bounded search decided as the whole walk on every lookup, and
    # visited fewer points.
    assert counters.get("restore.search_calls", 0) > 0, summary
    assert counters.get("restore.search_mismatches", 0) == 0, summary
    assert counters.get("restore.search_points", 0) < counters.get(
        "restore.search_points_whole", 0
    ), summary
    assert counters.get("restore.search_units", 0) < counters.get(
        "restore.search_units_whole", 0
    ), summary
    # Prometheus saw the connector's stats (the production path raises on a
    # key or label it does not know).
    assert lookups > 0 and seconds > 0 and admissions > 0 and searches > 0, summary
    # The logs.
    assert summaries, tiered_lines[-20:]
    assert any("restore started" in line for line in restore_lines), restore_lines
    assert any("restore admitted" in line for line in restore_lines), restore_lines
