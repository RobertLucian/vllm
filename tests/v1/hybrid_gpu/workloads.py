# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Seeded token-id workloads shaped like an agent's sessions.

A tool turn extends the previous turn's whole sequence, its output included.
A user turn extends the previous prompt only: a reasoning model's chat
template drops the tool loop's reasoning there, so the session parts from
its own past at the last prompt's end. Lengths are counted in blocks of the
engine under test, so one workload fits every model family.
"""

from __future__ import annotations

import heapq
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm import SamplingParams, TokensPrompt

if TYPE_CHECKING:
    from vllm import LLM

# Token ids below this are left alone (BOS, EOS, padding in most vocabularies).
_FIRST_ID = 3


def token_ids(seed: int, n: int, vocab_size: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(_FIRST_ID, vocab_size) for _ in range(n)]


@dataclass
class Result:
    prompt: list[int]
    output: list[int]
    # Prompt tokens the engine did not compute: the depth it resumed at.
    num_cached_tokens: int
    logprobs: list[dict[int, float]] | None

    @property
    def sequence(self) -> list[int]:
        return self.prompt + self.output


def greedy(max_tokens: int = 1, logprobs: int | None = None) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
        detokenize=False,
        logprobs=logprobs,
    )


def run(
    llm: LLM,
    prompts: list[list[int]],
    params: SamplingParams | None = None,
    priorities: list[int] | None = None,
    cold: bool = False,
) -> list[Result]:
    """Submit `prompts` together and wait for all of them. `cold`: read no
    cached KV (a full prefill, the reference for cached runs)."""
    params = params or greedy()
    if cold:
        params = params.clone()
        params.skip_reading_prefix_cache = True
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts],
        params,
        use_tqdm=False,
        priority=priorities,
    )
    results = []
    for prompt, out in zip(prompts, outputs):
        completion = out.outputs[0]
        logprobs = None
        if completion.logprobs is not None:
            logprobs = [
                {token: lp.logprob for token, lp in position.items()}
                for position in completion.logprobs
            ]
        results.append(
            Result(
                prompt, list(completion.token_ids), out.num_cached_tokens or 0, logprobs
            )
        )
    return results


@dataclass
class Session:
    """One agent's conversation; `next_*` build the next turn's prompt."""

    seed: int
    vocab_size: int
    block_size: int
    # Lengths are multiples of this: the engine's prefix-match unit, so a
    # prompt ends where a cache hit can land.
    unit: int = 1
    prompt: list[int] = field(default_factory=list)
    sequence: list[int] = field(default_factory=list)
    turns: int = 0

    @classmethod
    def start(
        cls,
        seed: int,
        vocab_size: int,
        block_size: int,
        num_blocks: float,
        shared_prefix: list[int] | None = None,
        unit: int = 1,
    ) -> Session:
        session = cls(seed, vocab_size, block_size, unit)
        prefix = list(shared_prefix or [])
        own = session._length(num_blocks) - len(prefix)
        session.prompt = prefix + token_ids(seed, max(own, unit), vocab_size)
        return session

    def _length(self, num_blocks: float) -> int:
        n = int(num_blocks * self.block_size)
        return max(self.unit, n - n % self.unit)

    def _new_tokens(self, num_blocks: float) -> list[int]:
        n = self._length(num_blocks)
        return token_ids(self.seed * 1_000 + self.turns + 1, n, self.vocab_size)

    def next_tool_turn(self, num_blocks: float) -> list[int]:
        """The whole sequence so far, then a tool result."""
        self.prompt = self.sequence + self._new_tokens(num_blocks)
        return self.prompt

    def next_user_turn(self, num_blocks: float) -> list[int]:
        """The last prompt, without the turn's output, then a user message."""
        self.prompt = self.prompt + self._new_tokens(num_blocks)
        return self.prompt

    def record(self, result: Result) -> None:
        assert result.prompt == self.prompt
        self.sequence = result.sequence
        self.turns += 1


@dataclass
class Turn:
    session: int
    # The shallowest point the turn may resume at: its session's last prompt
    # end less one resume unit (`Probe.resume_unit`).
    floor: int
    result: Result
    request_id: str = ""
    # time.monotonic() when the turn was submitted, its first token came
    # back, and it finished.
    submitted: float = 0.0
    first_token: float = 0.0
    finished: float = 0.0

    @property
    def ttft(self) -> float:
        return self.first_token - self.submitted

    @property
    def short_by(self) -> int:
        """Tokens of its session's past the turn computed beyond the one
        unit a resume may recompute."""
        return max(0, self.floor - self.result.num_cached_tokens)


def drive(
    llm: LLM,
    sessions: list[Session],
    seconds: float,
    turn_blocks: float,
    resume_unit: int,
    user_every: int = 0,
    max_blocks: float | None = None,
    restart_blocks: float = 4,
    max_tokens: int = 1,
    on_submit: Callable[[str, list[int], int], None] | None = None,
    on_step: Callable[[float], None] | None = None,
    think_s: tuple[float, float] | dict[int, tuple[float, float]] | None = None,
    seed: int = 0,
    priority: int = 0,
) -> list[Turn]:
    """Agents in a loop, out of phase like real ones: each session's next
    turn is submitted the moment its last one finishes, or `think_s`
    (seconds, drawn uniformly per turn from `seed`; by session index when a
    dict, none for sessions it leaves out) after it, until `seconds` have
    passed, so allocations keep arriving while copies run.
    Every `user_every`-th turn is a user turn; a session that would outgrow
    `max_blocks` ends, and a new one of `restart_blocks` (a new seed) takes
    its place. Each turn generates `max_tokens` tokens, at `priority`
    (production's agents send 10). `on_submit(request id, prompt, floor)`
    runs before a turn is added, `on_step(started)` after each engine step
    with the time it started."""
    engine = llm.llm_engine
    params = greedy(max_tokens)
    pending: dict[str, tuple[int, int, float]] = {}
    first_token: dict[str, float] = {}
    turns: list[Turn] = []
    generation = [0] * len(sessions)
    rng = random.Random(seed)
    # (when, session): next turns due after their agent's pause.
    due: list[tuple[float, int]] = []

    def submit(i: int) -> None:
        session = sessions[i]
        if (
            max_blocks is not None
            and session.turns
            and (
                len(session.sequence) + session._length(turn_blocks)
                > max_blocks * session.block_size
            )
        ):
            generation[i] += 1
            session = sessions[i] = Session.start(
                session.seed + 1_000_003 * generation[i],
                session.vocab_size,
                session.block_size,
                restart_blocks,
                unit=session.unit,
            )
        if session.turns == 0:
            prompt, floor = session.prompt, 0
        else:
            last = len(session.prompt)
            floor = max(0, last - last % resume_unit - resume_unit)
            if user_every and session.turns % user_every == 0:
                prompt = session.next_user_turn(turn_blocks)
            else:
                prompt = session.next_tool_turn(turn_blocks)
        request_id = f"s{i}g{generation[i]}-t{session.turns}"
        if on_submit is not None:
            on_submit(request_id, prompt, floor)
        engine.add_request(
            request_id,
            TokensPrompt(prompt_token_ids=prompt),
            params,
            priority=priority,
        )
        pending[request_id] = (i, floor, time.monotonic())

    deadline = time.monotonic() + seconds
    for i in range(len(sessions)):
        submit(i)
    while pending or due:
        now = time.monotonic()
        while due and due[0][0] <= now:
            _, i = heapq.heappop(due)
            if now < deadline:
                submit(i)
        if not pending:
            if due:
                time.sleep(min(0.01, max(0.0, due[0][0] - now)))
            continue
        started = time.monotonic()
        outputs = engine.step()
        now = time.monotonic()
        if on_step is not None:
            on_step(started)
        for out in outputs:
            if out.outputs and out.outputs[0].token_ids:
                first_token.setdefault(out.request_id, now)
            if not out.finished:
                continue
            i, floor, submitted = pending.pop(out.request_id)
            result = Result(
                list(out.prompt_token_ids),
                list(out.outputs[0].token_ids),
                out.num_cached_tokens or 0,
                None,
            )
            sessions[i].record(result)
            turns.append(
                Turn(
                    i,
                    floor,
                    result,
                    out.request_id,
                    submitted,
                    first_token.pop(out.request_id, now),
                    now,
                )
            )
            if time.monotonic() >= deadline:
                continue
            pause = think_s.get(i) if isinstance(think_s, dict) else think_s
            if pause is None:
                submit(i)
            else:
                heapq.heappush(due, (now + rng.uniform(*pause), i))
    return turns


def fillers(
    seed: int, count: int, num_blocks: float, vocab_size: int, block_size: int
) -> list[list[int]]:
    """Unrelated prompts that push older sessions out of the GPU."""
    n = int(num_blocks * block_size)
    return [
        token_ids(10_000_000 + seed * 1_000 + i, n, vocab_size) for i in range(count)
    ]
