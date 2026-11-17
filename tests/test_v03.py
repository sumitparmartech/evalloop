import re
import time

import pytest

from evalloop import (
    Case, Trace, Tracer, anthropic_agent, assert_reliable, diff_runs, from_anthropic_messages,
    from_openai_messages, judge, judge_with, load_traces, no_pii, replay_agent, run_suite, save_traces,
    tool_args, tool_count, to_junit, to_markdown, answer_contains,
)
from evalloop.stats import pass_at_k, pass_hat_k, two_proportion_p, wilson_interval


# ---------------------------------------------------------------- stats
def test_pass_at_k_and_pass_hat_k_match_known_values():
    assert pass_at_k(5, 3, 2) == pytest.approx(0.9)
    assert pass_hat_k(5, 3, 2) == pytest.approx(0.3)
    assert pass_at_k(5, 0, 3) == 0.0
    assert pass_hat_k(5, 5, 5) == 1.0
    assert pass_at_k(5, 4, 2) == 1.0


def test_wilson_interval_is_sensible():
    lo, hi = wilson_interval(5, 5)
    assert hi == 1.0 and 0.5 < lo < 0.7
    lo, hi = wilson_interval(0, 0)
    assert (lo, hi) == (0.0, 1.0)
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.25


def test_two_proportion_p_detects_real_and_ignores_noise():
    assert two_proportion_p(5, 5, 4, 5) > 0.5  # 1 failure in 5 is noise
    assert two_proportion_p(30, 30, 15, 30) < 0.01  # a big drop is real
    assert two_proportion_p(0, 0, 3, 5) == 1.0


# ---------------------------------------------------------------- runner
def flaky_agent(text, ctx):
    return "ok" if ctx.seed % 2 == 0 else "bad"


def test_flaky_detection_and_interval():
    res = run_suite(flaky_agent, [Case("c", "hi", [answer_contains("ok")])], trials=20)
    c = res.cases[0]
    assert c.flaky and "c" in res.flaky_cases
    lo, hi = c.interval
    assert lo < c.pass_rate < hi
    assert 0 <= c.pass_at(5) <= 1 and c.pass_hat(5) <= c.pass_rate


def test_parallel_run_matches_sequential_and_is_faster():
    def slow(text, ctx):
        time.sleep(0.05)
        return "ok"

    cases = [Case(f"c{i}", "x", [answer_contains("ok")]) for i in range(4)]
    t0 = time.time()
    seq = run_suite(slow, cases, trials=2, workers=1)
    t_seq = time.time() - t0
    t0 = time.time()
    par = run_suite(slow, cases, trials=2, workers=8)
    t_par = time.time() - t0
    assert [c.id for c in par.cases] == [c.id for c in seq.cases]
    assert par.pass_rate == seq.pass_rate == 1.0
    assert t_par < t_seq / 2


def test_timeout_marks_trial_failed_not_hung():
    def hang(text, ctx):
        time.sleep(1)
        return "ok"

    res = run_suite(hang, [Case("c", "x")], trials=1, timeout=0.1)
    assert not res.cases[0].trials[0].passed
    assert "did not finish" in res.cases[0].trials[0].trace.error


def test_diff_alpha_ignores_noise_but_catches_real_drops():
    def make(rate_pass, n):
        k = {"i": 0}

        def agent(text, ctx):
            k["i"] += 1
            return "ok" if k["i"] <= rate_pass else "bad"

        return run_suite(agent, [Case("c", "x", [answer_contains("ok")])], trials=n)

    base = make(5, 5)
    noisy = make(4, 5)  # 1 flaky failure
    assert diff_runs(base, noisy).has_regressions  # strict default still flags it
    d = diff_runs(base, noisy, alpha=0.2)
    assert not d.has_regressions and d.unchanged[0]["noise"]
    big_base, big_bad = make(30, 30), make(12, 30)
    assert diff_runs(big_base, big_bad, alpha=0.05).has_regressions


# ---------------------------------------------------------------- checks
def test_tool_args_predicate_and_regex():
    t = Tracer("q")
    t.tool("refund", {"amount": 40, "note": "Late delivery"}, output={})
    tr = t.finish("done")
    case = Case("c", "q")
    assert tool_args("refund", amount=lambda x: x <= 50)(tr, case).passed
    assert not tool_args("refund", amount=lambda x: x > 50)(tr, case).passed
    assert tool_args("refund", note=re.compile("late", re.I))(tr, case).passed


def test_no_pii_flags_and_allows():
    case = Case("c", "q")
    bad = Trace("q", answer="Email me at jane.doe@example.com or call 555-123-4567")
    ok = Trace("q", answer="Your refund is on its way.")
    assert not no_pii()(bad, case).passed
    assert no_pii()(ok, case).passed
    assert no_pii("email", allow=("jane.doe@example.com",))(bad, case).passed
    assert not no_pii("api_key")(Trace("q", answer="key sk-abcdefghijklmnopqrstuv"), case).passed


def test_tool_count_limit():
    t = Tracer("q")
    for _ in range(3):
        t.tool("search", {"q": 1}, output="r")
    tr = t.finish("x")
    assert tool_count("search", 3)(tr, Case("c", "q")).passed
    assert not tool_count("search", 2)(tr, Case("c", "q")).passed


def test_judge_parses_score_and_handles_garbage():
    tr = Trace("help", answer="Sure, here you go.")
    case = Case("c", "help")
    good = judge("is polite", ask=lambda p: 'Sure {"score": 5, "reason": "polite"}')
    low = judge("is polite", ask=lambda p: '{"score": 2, "reason": "rude"}')
    junk = judge("is polite", ask=lambda p: "no idea")
    seen = {}
    judge("c", ask=lambda p: seen.setdefault("p", p) and '{"score": 5}')(tr, case)
    assert good(tr, case).passed and "5/5" in good(tr, case).detail
    assert not low(tr, case).passed
    assert not junk(tr, case).passed
    assert "Agent answer: Sure, here you go." in seen["p"]


# ---------------------------------------------------------------- adapters
def test_anthropic_agent_runs_tool_loop_and_hides_key():
    sent = []
    replies = [
        {"content": [{"type": "tool_use", "id": "t1", "name": "lookup", "input": {"id": 7}}],
         "usage": {"input_tokens": 10, "output_tokens": 5}},
        {"content": [{"type": "text", "text": "Order 7 is shipped."}],
         "usage": {"input_tokens": 20, "output_tokens": 6}},
    ]

    def fake_post(url, payload, headers, timeout):
        sent.append((url, payload, headers))
        return replies[len(sent) - 1]

    agent = anthropic_agent(
        system="s", tools=[{"name": "lookup", "description": "d", "input_schema": {"type": "object"}}],
        impls={"lookup": lambda id: {"status": "shipped"}}, model="m", api_key="SECRET-KEY", post=fake_post,
    )
    tr = agent("where is order 7?")
    assert tr.answer == "Order 7 is shipped."
    assert [s.name for s in tr.tool_calls] == ["lookup"] and tr.tool_calls[0].output == {"status": "shipped"}
    assert tr.tokens == 41
    assert sent[0][2]["x-api-key"] == "SECRET-KEY" and "SECRET-KEY" not in str(sent[0][1])
    # the second request carries the tool_result back
    assert sent[1][1]["messages"][-1]["content"][0]["type"] == "tool_result"


def test_anthropic_agent_gives_up_after_max_turns():
    loop = {"content": [{"type": "tool_use", "id": "t", "name": "x", "input": {}}]}
    agent = anthropic_agent(system="s", tools=[], impls={"x": lambda: 1}, model="m",
                            max_turns=2, post=lambda *a: loop)
    assert "no final answer" in agent("hi").error


def test_judge_with_openai_style_and_anthropic():
    ask = judge_with("groq", post=lambda url, p, h, t: {"choices": [{"message": {"content": "hello"}}]}, api_key="k")
    assert ask("q") == "hello"
    ask = judge_with("anthropic", post=lambda url, p, h, t: {"content": [{"type": "text", "text": "hey"}]}, api_key="k")
    assert ask("q") == "hey"
    with pytest.raises(ValueError):
        judge_with("nope")


# ---------------------------------------------------------------- traces
def test_import_openai_messages():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "refund order 5"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "function": {"name": "refund", "arguments": '{"id": 5}'}}]},
        {"role": "tool", "tool_call_id": "a", "content": '{"error": "denied"}'},
        {"role": "assistant", "content": "Sorry, denied."},
    ]
    tr = from_openai_messages(msgs)
    assert tr.input == "refund order 5" and tr.answer == "Sorry, denied."
    assert tr.tool_calls[0].input == {"id": 5} and tr.tool_calls[0].error == "denied"


def test_import_anthropic_messages():
    msgs = [
        {"role": "user", "content": "status of 9?"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "u1", "name": "lookup", "input": {"id": 9}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1", "content": '{"status": "ok"}'}]},
        {"role": "assistant", "content": [{"type": "text", "text": "It is ok."}]},
    ]
    tr = from_anthropic_messages(msgs)
    assert tr.input == "status of 9?" and tr.answer == "It is ok."
    assert tr.tool_calls[0].output == {"status": "ok"}


def test_save_load_and_replay_traces(tmp_path):
    t1 = Trace("a", answer="A")
    t2 = Trace("b", answer="B")
    p = str(tmp_path / "t.jsonl")
    assert save_traces(p, [t1, t2]) == 2
    loaded = load_traces(p)
    res = run_suite(replay_agent(loaded), [Case("a", "a", [answer_contains("A")]), Case("b", "b", [answer_contains("X")])])
    assert res.cases[0].pass_all and not res.cases[1].pass_all
    missing = run_suite(replay_agent(loaded), [Case("z", "z")])
    assert "no recorded trace" in missing.cases[0].trials[0].trace.error


# ---------------------------------------------------------------- ci
def test_junit_and_markdown_and_assert_reliable():
    good = Case("good", "x", [answer_contains("ok")])
    bad = Case("bad & <odd>", "y", [answer_contains("nope")])
    res = run_suite(lambda t, c: "ok", [good, bad], trials=3, suite="s")
    xml = to_junit(res)
    assert 'tests="2" failures="1"' in xml and "&amp;" in xml and "&lt;odd&gt;" in xml
    import xml.dom.minidom as md
    md.parseString(xml)  # valid XML
    md_text = to_markdown(res)
    assert "Reliability 50%" in md_text and "bad & <odd>" in md_text.replace("`", "")
    assert_reliable(lambda t, c: "ok", good, trials=3)
    with pytest.raises(AssertionError, match="passed 0/3"):
        assert_reliable(lambda t, c: "ok", bad, trials=3)
