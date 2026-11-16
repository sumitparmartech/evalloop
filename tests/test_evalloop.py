import json
import os
import sys

import pytest

from evalloop import (
    Case, SuiteResult, Tracer, answer_contains, answer_not_contains, cost_under, custom,
    diff_runs, grounded, json_answer, latency_under, max_steps, no_loop, no_secret_leak,
    render_html, run_suite, tool_args, tool_called, tool_not_called, tool_order,
)
from evalloop.cli import main

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from suite import cases as demo_cases  # noqa: E402
from support_agent import agent_v1, agent_v2  # noqa: E402


def make_trace():
    t = Tracer("refund order 7 for $20")
    t.tool("lookup", {"id": 7}, output={"total": 20.0}, latency_ms=100)
    t.tool("refund", {"id": 7, "amount": 20.0}, output={"ok": True}, latency_ms=50)
    t.llm("answer", "p", "o", tokens_in=1000, tokens_out=100, latency_ms=200)
    return t.finish("Refunded $20.00 for order 7")


def run(check, trace=None):
    return check(trace or make_trace(), None)


def test_tool_checks():
    assert run(tool_called("lookup")).passed
    assert run(tool_called("lookup", times=1)).passed
    assert not run(tool_called("lookup", times=2)).passed
    assert run(tool_not_called("delete")).passed
    assert run(tool_order("lookup", "refund")).passed
    assert not run(tool_order("refund", "lookup")).passed
    assert run(tool_args("refund", id=7, amount=20)).passed
    assert not run(tool_args("refund", amount=25)).passed
    assert not run(tool_args("missing")).passed


def test_answer_checks():
    assert run(answer_contains("refunded", "order 7")).passed
    assert run(answer_contains("zzz", "refunded", any_of=True)).passed
    assert not run(answer_contains("zzz")).passed
    assert run(answer_not_contains("denied")).passed
    assert not run(answer_not_contains("refunded")).passed


def test_json_answer():
    t = Tracer("x")
    assert run(json_answer("a", "b"), t.finish('{"a": 1, "b": 2}')).passed
    assert not run(json_answer("a", "c"), Tracer("x").finish('{"a": 1}')).passed
    assert not run(json_answer("a"), Tracer("x").finish("not json")).passed


def test_grounded_catches_invented_numbers():
    assert run(grounded()).passed
    bad = make_trace()
    bad.answer = "Refunded $30.00 for order 7"
    res = run(grounded(), bad)
    assert not res.passed and "30.0" in res.detail


def test_secret_leak():
    t = Tracer("x").finish("the key is SK-123")
    assert not run(no_secret_leak("sk-123"), t).passed
    assert run(no_secret_leak("sk-999"), t).passed


def test_loop_detection():
    t = Tracer("x")
    for _ in range(4):
        t.tool("search", {"q": "a"}, output="none", latency_ms=1)
    t.finish("nothing")
    assert not run(no_loop(2), t.trace).passed
    assert run(no_loop(4), t.trace).passed


def test_budgets_and_steps():
    tr = make_trace()
    assert tr.latency_ms == 350
    assert run(latency_under(400)).passed and not run(latency_under(300)).passed
    assert abs(tr.cost - (1000 * 2.5 / 1e6 + 100 * 10 / 1e6)) < 1e-9
    assert run(cost_under(0.01)).passed and not run(cost_under(0.001)).passed
    assert run(max_steps(3)).passed and not run(max_steps(2)).passed


def test_check_exception_is_a_failure():
    def boom(trace, case):
        raise ValueError("nope")

    res = run(custom("boom", boom))
    assert not res.passed and "ValueError" in res.detail


def test_runner_is_deterministic_and_handles_crashes():
    def agent(text, ctx):
        if ctx.trial == 1:
            raise RuntimeError("model timeout")
        return Tracer(text).finish("ok")

    cases = [Case("c", "hi", [answer_contains("ok")])]
    res = run_suite(agent, cases, trials=3)
    c = res.cases[0]
    assert [t.passed for t in c.trials] == [True, False, True]
    assert c.pass_any and not c.pass_all
    assert "RuntimeError" in c.trials[1].trace.error
    again = run_suite(agent, cases, trials=3)
    assert [t.passed for t in again.cases[0].trials] == [True, False, True]


def test_plain_string_agent_is_accepted():
    res = run_suite(lambda text, ctx: "hello", [Case("c", "x", [answer_contains("hello")])])
    assert res.pass_rate == 1.0


def test_demo_v2_beats_v1_and_diff_finds_it():
    v1 = run_suite(agent_v1, demo_cases, trials=5, label="v1", seed=2)
    v2 = run_suite(agent_v2, demo_cases, trials=5, label="v2", seed=2)
    assert v2.pass_rate == 1.0 and v2.reliability == 1.0
    assert v1.pass_rate < 0.6 and v1.reliability < 0.5
    d = diff_runs(v1, v2)
    assert not d.has_regressions and len(d.improved) >= 4
    back = diff_runs(v2, v1)
    assert back.has_regressions
    assert v2.cost_per_run < v1.cost_per_run and v2.latency_p95 < v1.latency_p95


def test_save_load_roundtrip(tmp_path):
    res = run_suite(agent_v1, demo_cases, trials=2, label="x", seed=2)
    p = tmp_path / "run.json"
    res.save(str(p))
    loaded = SuiteResult.load(str(p))
    assert loaded.summary() == pytest.approx(res.summary())
    assert json.loads(p.read_text())["summary"]["pass_rate"] == pytest.approx(res.pass_rate)


def test_report_contains_cases_and_escapes_html():
    cases = [Case("<script>x</script>", "a & b", [answer_contains("zzz")])]
    res = run_suite(lambda text, ctx: "<b>hi</b>", cases, trials=2)
    page = render_html(res)
    assert "<script>x</script>" not in page and "&lt;script&gt;" in page
    assert "<b>hi</b>" not in page
    v2 = run_suite(agent_v2, demo_cases, trials=2, seed=2)
    v1 = run_suite(agent_v1, demo_cases, trials=2, seed=2, label="v1")
    page = render_html(v2, v1)
    assert "improved" in page and "refund_small_order" in page


def test_cli_exit_codes(tmp_path):
    base = tmp_path / "v1.json"
    new = tmp_path / "v2.json"
    cwd = os.getcwd()
    os.chdir(os.path.join(os.path.dirname(__file__), "..", "examples"))
    try:
        assert main(["run", "suite:cases", "--agent", "support_agent:agent_v2", "--trials", "2", "--out", str(new), "--seed", "2"]) == 0
        assert main(["run", "suite:cases", "--agent", "support_agent:agent_v1", "--trials", "2", "--out", str(base), "--seed", "2", "--fail-under", "0.9"]) == 1
        assert main(["diff", str(new), str(base), "--fail-on-regression"]) == 1
        assert main(["diff", str(base), str(new), "--fail-on-regression"]) == 0
        html = tmp_path / "r.html"
        assert main(["report", str(new), "--baseline", str(base), "--html", str(html)]) == 0
        assert html.read_text().startswith("<!doctype html>")
    finally:
        os.chdir(cwd)
