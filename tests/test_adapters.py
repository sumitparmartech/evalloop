import json

import pytest

from evalloop import AdapterError, Case, chat_agent, from_provider, grounded, run_suite, tool_order
from evalloop.adapters import PROVIDERS


def tc(name, args, id="c1"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def reply(content=None, calls=None, pt=40, ct=10):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = calls
    return {"choices": [{"message": msg}], "usage": {"prompt_tokens": pt, "completion_tokens": ct}}


class Fake:
    """Scripted provider that records every request."""

    def __init__(self, script):
        self.script, self.seen = list(script), []

    def __call__(self, url, payload, headers, timeout):
        self.seen.append((url, payload, headers))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


IMPLS = {"lookup_order": lambda order_id: {"total": 49.0}, "issue_refund": lambda order_id, amount: {"ok": True}}


def make(fake, **kw):
    return chat_agent(system="s", tools=[], impls=IMPLS, model="m", base_url="https://x.test/v1/", api_key="SECRET-KEY", post=fake, sleep=lambda s: None, **kw)


def test_tool_loop_is_traced():
    fake = Fake([reply(calls=[tc("lookup_order", {"order_id": "1042"})]), reply(calls=[tc("issue_refund", {"order_id": "1042", "amount": 49.0}, "c2")]), reply("Refunded $49.00.")])
    trace = make(fake)("refund 1042", None)
    assert [s.name for s in trace.steps] == ["plan", "lookup_order", "plan", "issue_refund", "answer"]
    assert trace.answer == "Refunded $49.00."
    assert trace.steps[0].tokens_in == 40 and trace.steps[0].tokens_out == 10
    assert fake.seen[0][0] == "https://x.test/v1/chat/completions"
    assert fake.seen[0][2]["Authorization"] == "Bearer SECRET-KEY"


def test_works_with_checks_and_run_suite():
    def factory():
        return make(Fake([reply(calls=[tc("lookup_order", {"order_id": "1042"})]), reply(calls=[tc("issue_refund", {"order_id": "1042", "amount": 49.0}, "c2")]), reply("Refunded $49.00.")]))
    res = run_suite(factory(), [Case("c", "refund 1042", [tool_order("lookup_order", "issue_refund"), grounded()])], trials=1)
    assert res.cases[0].pass_all


def test_tool_error_is_fed_back_not_raised():
    impls = {"lookup_order": lambda order_id: 1 / 0}
    fake = Fake([reply(calls=[tc("lookup_order", {"order_id": "1"})]), reply("Sorry, that failed.")])
    agent = chat_agent(system="s", tools=[], impls=impls, model="m", api_key="k", post=fake, sleep=lambda s: None)
    trace = agent("x", None)
    assert trace.steps[1].error and "ZeroDivisionError" in trace.steps[1].error
    assert "error" in json.dumps(fake.seen[1][1]["messages"][-1])


def test_unknown_tool_and_max_turns():
    fake = Fake([reply(calls=[tc("nope", {})]) for _ in range(3)])
    trace = make(fake, max_turns=3)("x", None)
    assert trace.error and "3 turns" in trace.error
    assert trace.steps[1].output == {"error": "unknown tool nope"}


def test_retries_on_429_then_succeeds():
    waits = []
    fake = Fake([AdapterError("HTTP 429", 429), reply("ok")])
    agent = chat_agent(system="s", tools=[], impls={}, model="m", api_key="k", post=fake, sleep=waits.append)
    assert agent("x", None).answer == "ok" and waits == [2]


def test_error_never_contains_the_key():
    fake = Fake([AdapterError("HTTP 401: invalid api key", 401)])
    with pytest.raises(AdapterError) as e:
        make(fake)("x", None)
    assert "SECRET-KEY" not in str(e.value) and e.value.status == 401


def test_key_comes_from_env_and_ollama_needs_none(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    fake = Fake([reply("hi")])
    from_provider("groq", system="s", tools=[], impls={}, post=fake)("x", None)
    assert fake.seen[0][2]["Authorization"] == "Bearer from-env"
    assert fake.seen[0][0].startswith("https://api.groq.com/openai/v1")
    fake2 = Fake([reply("hi")])
    from_provider("ollama", system="s", tools=[], impls={}, post=fake2)("x", None)
    assert "Authorization" not in fake2.seen[0][2]
    with pytest.raises(ValueError):
        from_provider("nope", system="s", tools=[], impls={})
    assert "groq" in PROVIDERS
