import pytest

from evalloop import (
    Case, Conversation, Trace, Turn, answer_contains, anthropic_agent, chat_agent, grounded, no_pii,
    no_secret_leak, run_suite, tool_count, SuiteResult, Tracer,
)


def echo_memory_agent(text, ctx):
    """Answers with the first thing the user ever said, so history matters."""
    first = ctx.history[0][0] if ctx.history else text
    return Tracer(text).finish(f"you first said: {first}")


def test_history_and_turn_index_reach_the_agent():
    seen = []

    def agent(text, ctx):
        seen.append((ctx.turn, list(ctx.history)))
        return f"a{ctx.turn}"

    res = run_suite(agent, [Conversation("c", ["one", "two", "three"])], trials=1)
    assert [t for t, _ in seen] == [0, 1, 2]
    assert seen[2][1] == [("one", "a0"), ("two", "a1")]
    tr = res.cases[0].trials[0].trace
    assert tr.turn_inputs == ["one", "two", "three"] and tr.turn_answers == ["a0", "a1", "a2"] and tr.answer == "a2"


def test_per_turn_checks_and_whole_conversation_checks():
    case = Conversation(
        "c",
        ["hello", Turn("again", [answer_contains("first said: hello")])],
        [answer_contains("hello")],
    )
    ok = run_suite(echo_memory_agent, [case]).cases[0].trials[0]
    assert ok.passed
    names = [c.name for c in ok.checks]
    assert any(n.startswith("turn 2:") for n in names)

    forgetful = run_suite(lambda t, c: "no idea", [case]).cases[0].trials[0]
    assert not forgetful.passed
    assert any(c.name.startswith("turn 2:") and not c.passed for c in forgetful.checks)


def test_scripted_user_can_react_to_previous_answer():
    case = Conversation("c", ["ping", Turn(lambda h: "got " + h[-1][1])])
    tr = run_suite(lambda t, c: "pong" if t == "ping" else "ok", [case]).cases[0].trials[0].trace
    assert tr.turn_inputs == ["ping", "got pong"]


def test_error_on_a_turn_stops_the_conversation_and_fails_it():
    def agent(text, ctx):
        if ctx.turn == 1:
            raise RuntimeError("boom")
        return "fine"

    res = run_suite(agent, [Conversation("c", ["a", "b", "c"], [answer_contains("fine")])])
    t = res.cases[0].trials[0]
    assert not t.passed
    assert t.trace.turn_inputs == ["a", "b"]  # never reached turn 3
    assert "boom" in t.trace.error and t.checks[-1].name == "turn 2: agent_completed"


def test_tool_budget_is_counted_across_all_turns():
    def agent(text, ctx):
        t = Tracer(text)
        t.tool("search", {"q": text}, output="r")
        return t.finish("ok")

    case = Conversation("c", ["a", "b", "c"], [tool_count("search", at_most=2)])
    assert not run_suite(agent, [case]).cases[0].trials[0].passed
    assert run_suite(agent, [Conversation("c", ["a", "b"], [tool_count("search", 2)])]).cases[0].trials[0].passed


def test_secret_and_pii_are_checked_in_every_answer_not_just_the_last():
    answers = iter(["The key is sk-demo-9f3a1c77", "Sorry, cannot help.", "Mail me at a.b@x.com"])
    agent = lambda t, c: next(answers)
    case = Conversation("c", ["1", "2", "3"], [no_secret_leak("sk-demo-9f3a1c77")])
    assert not run_suite(agent, [case]).cases[0].trials[0].passed
    answers = iter(["Contact a.b@x.com", "ok", "ok"])
    case = Conversation("c", ["1", "2", "3"], [no_pii("email")])
    assert not run_suite(agent, [case]).cases[0].trials[0].passed


def test_grounded_knows_numbers_from_earlier_turns():
    def agent(text, ctx):
        t = Tracer(text)
        if ctx.turn == 0:
            t.tool("lookup", {"id": 1}, output={"total": 49})
            return t.finish("It costs 49.")
        return t.finish("As I said, 49.")  # 49 came from a tool in turn 1 and is still grounded

    case = Conversation("c", ["price?", Turn("again?", [grounded()])])
    assert run_suite(agent, [case]).cases[0].trials[0].passed

    def liar(text, ctx):
        return Tracer(text).finish("It costs 77.")

    assert not run_suite(liar, [Conversation("c", ["price?", Turn("again?", [grounded()])])]).cases[0].trials[0].passed


def test_conversations_work_in_parallel_and_survive_save_load(tmp_path):
    case = Conversation("c", ["a", Turn("b", [answer_contains("first said: a")])])
    res = run_suite(echo_memory_agent, [case] * 1, trials=4, workers=4)
    assert res.pass_rate == 1.0
    path = str(tmp_path / "r.json")
    res.save(path)
    back = SuiteResult.load(path)
    assert back.cases[0].trials[0].trace.turn_answers == ["you first said: a", "you first said: a"]


def test_plain_cases_are_unchanged():
    res = run_suite(lambda t, c: "ok", [Case("c", "x", [answer_contains("ok")])], trials=2)
    assert res.pass_rate == 1.0 and res.cases[0].trials[0].trace.turn_answers == []


def test_adapters_send_history_to_the_model():
    sent = []

    def post(url, payload, headers, timeout):
        sent.append(payload)
        return {"choices": [{"message": {"content": "done"}}], "content": [{"type": "text", "text": "done"}]}

    case = Conversation("c", ["first q", "second q"])
    run_suite(chat_agent(system="s", tools=[], impls={}, model="m", post=post, api_key="k"), [case])
    assert [m["role"] for m in sent[1]["messages"]] == ["system", "user", "assistant", "user"]
    assert sent[1]["messages"][1]["content"] == "first q"

    sent.clear()
    run_suite(anthropic_agent(system="s", tools=[], impls={}, model="m", post=post, api_key="k"), [case])
    assert [m["role"] for m in sent[1]["messages"]] == ["user", "assistant", "user"]
