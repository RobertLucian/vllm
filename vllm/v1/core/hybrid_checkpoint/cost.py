# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The prefill cost that GreedyDual-Size eviction saves by keeping a unit."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefillCostModel:
    """Seconds to prefill positions `[a, b)`: `per_token` seconds per token
    plus `per_token_per_100k` seconds per token per 100,000 tokens of
    position, the attention term that grows with context.

    With a slower KV tier under the GPU pool (a KV connector's CPU pool),
    `reload_per_token` is the seconds per token to load a unit back from it
    instead; None prefills every evicted unit again.

    GreedyDual-Size ranks units by `stamp + gain / size` with `stamp`
    itself a past score, so scaling every coefficient by a constant does not
    change the eviction order; only their ratios matter.
    """

    per_token: float
    per_token_per_100k: float
    reload_per_token: float | None = None

    def seconds(self, a: int, b: int) -> float:
        if b <= a:
            return 0.0
        return (b - a) * self.per_token + self.per_token_per_100k * (
            b * b - a * a
        ) / 2e5

    def reload_seconds(self, a: int, b: int) -> float:
        assert self.reload_per_token is not None
        return max(0, b - a) * self.reload_per_token
