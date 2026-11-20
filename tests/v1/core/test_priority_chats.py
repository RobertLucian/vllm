# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The priority policy and urgent requests (priority 0 or less: chats beside
agents): running requests in priority order, a cap on the others' prefill
while an urgent request waits or prefills, urgent prefill on the steps
`prefill_schedule_interval` defers, their own long-prefill threshold, and
running slots kept for them."""

import pytest

from vllm.config import SchedulerConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from .test_scheduler import (
    create_requests_with_priority,
    create_scheduler_with_priority,
)

pytestmark = pytest.mark.cpu_test

AGENT, CHAT = 10, 0


def make(
    *,
    prefill_cap: int = 0,
    reserved: int = 0,
    urgent_threshold: int = 0,
    budget: int = 256,
    max_num_seqs: int = 16,
) -> Scheduler:
    """Chunks of at most 128 tokens, `budget` tokens a step."""
    scheduler = create_scheduler_with_priority(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=budget,
        long_prefill_token_threshold=128,
        max_model_len=2048,
    )
    scheduler.low_priority_prefill_tokens = prefill_cap
    scheduler.reserved_priority_slots = reserved
    scheduler.urgent_long_prefill_threshold = urgent_threshold
    return scheduler


def add(
    scheduler: Scheduler, req_id: str, priority: int, num_tokens: int, arrival: float
) -> Request:
    request = create_requests_with_priority(
        1, [priority], [arrival], num_tokens=num_tokens, req_ids=[req_id]
    )[0]
    scheduler.add_request(request)
    return request


def step(scheduler: Scheduler, *, throttle: bool = False) -> dict[str, int]:
    """One step; a request whose prompt is done samples a token."""
    output = scheduler.schedule(throttle)
    req_ids = list(output.num_scheduled_tokens)
    sampled = [
        [1]
        if scheduler.requests[r].num_computed_tokens >= scheduler.requests[r].num_tokens
        else []
        for r in req_ids
    ]
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return dict(output.num_scheduled_tokens)


def test_running_requests_are_scheduled_in_priority_order():
    """A chat admitted after two agents runs ahead of them."""
    scheduler = make(budget=512)
    add(scheduler, "a1", AGENT, 1000, 1.0)
    add(scheduler, "a2", AGENT, 1000, 2.0)
    step(scheduler)
    add(scheduler, "chat", CHAT, 300, 3.0)
    assert step(scheduler) == {"a1": 128, "a2": 128, "chat": 128}
    assert [r.request_id for r in scheduler.running] == ["chat", "a1", "a2"]


@pytest.mark.parametrize("prefill_cap", [0, 128])
def test_agents_share_a_prefill_cap_while_a_chat_prefills(prefill_cap: int):
    """Two agents' chunks fill the step: without a cap the chat waits for
    their prefills. With one, the agents share 128 prefill tokens while the
    chat waits or prefills, and get their chunks back once it decodes."""
    scheduler = make(prefill_cap=prefill_cap)
    add(scheduler, "a1", AGENT, 1000, 1.0)
    add(scheduler, "a2", AGENT, 1000, 2.0)
    step(scheduler)
    add(scheduler, "chat", CHAT, 300, 3.0)
    if not prefill_cap:
        assert step(scheduler) == {"a1": 128, "a2": 128}
        return
    assert step(scheduler) == {"a1": 128, "chat": 128}
    assert step(scheduler) == {"chat": 128, "a1": 128}
    # The chat's last 44 prompt tokens, then its first decode.
    assert step(scheduler) == {"chat": 44, "a1": 128}
    assert step(scheduler) == {"chat": 1, "a1": 128, "a2": 127}


def test_a_waiting_chats_chunk_comes_out_of_the_step_first():
    """The running agents are scheduled before the waiting chat: a cap of 200
    alone would leave the chat 56 of the step's 256 tokens. Its 128-token
    chunk is kept for it, so the agents share what is left."""
    scheduler = make(prefill_cap=200)
    add(scheduler, "a1", AGENT, 1000, 1.0)
    add(scheduler, "a2", AGENT, 1000, 2.0)
    step(scheduler)
    add(scheduler, "chat", CHAT, 300, 3.0)
    assert step(scheduler) == {"a1": 128, "chat": 128}


def test_a_chat_has_its_own_long_prefill_threshold():
    scheduler = make(prefill_cap=128, urgent_threshold=256, budget=512)
    add(scheduler, "a1", AGENT, 1000, 1.0)
    step(scheduler)
    add(scheduler, "chat", CHAT, 600, 2.0)
    assert step(scheduler) == {"a1": 128, "chat": 256}


@pytest.mark.parametrize("prefill_cap", [0, 128])
def test_a_chat_prefills_on_steps_that_defer_prefill(prefill_cap: int):
    """`prefill_schedule_interval` defers prefill to every Nth step; an
    agent's chunk waits for one, a chat's does not while the cap is on."""
    scheduler = make(prefill_cap=prefill_cap, budget=512)
    add(scheduler, "decoding", AGENT, 16, 1.0)
    add(scheduler, "a1", AGENT, 1000, 2.0)
    step(scheduler)
    add(scheduler, "chat", CHAT, 300, 3.0)
    scheduled = step(scheduler, throttle=True)
    assert scheduled == (
        {"decoding": 1, "chat": 128} if prefill_cap else {"decoding": 1}
    )


@pytest.mark.parametrize("reserved", [0, 1])
def test_running_slots_kept_for_chats(reserved: int):
    """Four running slots, one kept for chats: three agents run and a chat
    that arrives later takes the fourth."""
    scheduler = make(reserved=reserved, max_num_seqs=4, budget=1024)
    for i in range(5):
        add(scheduler, f"a{i}", AGENT, 100, float(i))
    step(scheduler)
    assert len(scheduler.running) == 4 - reserved
    add(scheduler, "chat", CHAT, 100, 9.0)
    scheduled = step(scheduler)
    assert ("chat" in scheduled) == bool(reserved)
    assert len(scheduler.running) == 4


def test_every_slot_reserved_is_refused():
    with pytest.raises(ValueError, match="reserved_priority_slots"):
        SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=2048,
            max_model_len=2048,
            is_encoder_decoder=False,
            reserved_priority_slots=4,
        )
