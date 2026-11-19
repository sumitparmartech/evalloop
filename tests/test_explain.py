"""explain_case: find why a flaky case fails by comparing its passing and failing trials."""
import json

from evalloop import (
    Case, Tracer, answer_contains, explain_case, explain_suite, max_steps, render_html, run_suite,
    tool_args, tool_called, tool_not_called,
)
from evalloop.cli import main

BAD_TRIALS = {1, 4}  # these trials take the bad path; the rest pass


def _agent(behaviour):
    def agent(text, ctx):
        t = Tracer(text)
        bad = ctx.trial in BAD_TRIALS
        behaviour(t, bad)
        return t.finish("done: escalated" if not bad or behaviour.__name__ != "answer_only" else "done")
    return agent


def skipped(t, bad):
    t.tool("lookup_order", {"order_id": "1044"}, output={"total": 899}, latency_ms=50)
    if bad:
        t.tool("issue_refund", {"order_id": "1044", "amount": 899}, output={"ok": True}, latency_ms=40)
    else:
        t.tool("escalate", {"order_id": "1044"}, output={"ticket": 7}, latency_ms=40)


def wrong_arg(t, bad):
    t.tool("lookup_order", {"order_id": "1043" if not bad else "1034"}, output={}, latency_ms=50)


def looping(t, bad):
    for _ in range(6 if bad else 1):
        t.tool("search_kb", {"q": "refund"}, output="...", latency_ms=20)


def erroring(t, bad):
    def flaky_tool(order_id):
        if bad:
            raise TimeoutError("upstream timed out")
        return {"status": "shipped"}
    t.tool("lookup_order", {"order_id": "1043"}, fn=flaky_tool)


def reordered(t, bad):
    first, second = ("issue_refund", "lookup_order") if bad else ("lookup_order", "issue_refund")
    t.tool(first, {"order_id": "7"}, output={}, latency_ms=10)
    t.tool(second, {"order_id": "7"}, output={}, latency_ms=10)


def answer_only(t, bad):
    t.tool("lookup_order", {"order_id": "1043"}, output={"status": "shipped"}, latency_ms=10)


def _explain(behaviour, checks, trials=6):
    res = run_suite(_agent(behaviour), [Case("c", "help", checks)], trials=trials)
    return res, explain_case(res.cases[0])


def kinds(ex):
    return [f.kind for f in ex.findings]


def test_all_pass_returns_none():
    res = run_suite(_agent(skipped), [Case("c", "help", [tool_called("lookup_order")])], trials=4)
    assert explain_case(res.cases[0]) is None
    assert explain_suite(res) == []


def test_skipped_and_extra_tool_are_found_with_counts():
    _, ex = _explain(skipped, [tool_called("escalate"), tool_not_called("issue_refund")])
    assert (ex.passes, ex.total) == (4, 6)
    assert kinds(ex)[:2] == ["failed_check", "failed_check"]
    skip = next(f for f in ex.findings if f.kind == "skipped_tool")
    assert skip.summary == "skipped escalate"
    assert (skip.fail_hits, skip.fail_total, skip.pass_hits, skip.pass_total) == (2, 2, 0, 4)
    assert skip.strength == 1.0
    assert any(f.kind == "extra_tool" and "issue_refund" in f.summary for f in ex.findings)
    assert ex.headline in ("skipped escalate", "called issue_refund, which passing runs do not call")
    d = ex.divergences[0]
    assert d.step == 1 and d.op == "changed" and d.expected.startswith("escalate(") and d.got.startswith("issue_refund(")
    assert [r.op for r in ex.alignment] == ["same", "changed", "same"]
    assert "likely cause" in ex.text() and "first split" in ex.text()


def test_wrong_argument_value():
    _, ex = _explain(wrong_arg, [tool_args("lookup_order", order_id="1043")])
    f = next(f for f in ex.findings if f.kind == "different_args")
    assert "order_id" in f.summary and "1034" in f.summary and "1043" in f.summary
    assert f.fail_hits == 2


def test_loop_shows_up_as_more_steps():
    _, ex = _explain(looping, [max_steps(3)])
    f = next(f for f in ex.findings if f.kind == "more_steps")
    assert "median 6 vs 1" in f.summary


def test_tool_error():
    _, ex = _explain(erroring, [custom_no_error()])
    f = next(f for f in ex.findings if f.kind == "tool_error")
    assert "TimeoutError" in f.summary and f.fail_hits == 2 and f.pass_hits == 0


def custom_no_error():
    from evalloop import custom
    return custom("no_tool_errors", lambda tr: all(not s.error for s in tr.steps))


def test_order():
    from evalloop import tool_order
    _, ex = _explain(reordered, [tool_order("lookup_order", "issue_refund")])
    assert any(f.kind == "order" for f in ex.findings)


def test_same_tools_points_at_the_answer():
    _, ex = _explain(answer_only, [answer_contains("escalated")])
    assert any(f.kind == "answer_only" for f in ex.findings)
    assert ex.headline.startswith("check answer_contains(escalated) failed")
    assert ex.alignment[-1].op == "changed"


def test_always_failing_describes_the_pattern_without_fake_causes():
    def agent(text, ctx):
        t = Tracer(text)
        for _ in range(3):
            t.tool("search_kb", {"q": "x"}, output="", latency_ms=5)
        return t.finish("?")
    res = run_suite(agent, [Case("loop", "help", [max_steps(2)])], trials=3)
    ex = explain_case(res.cases[0])
    assert ex.passes == 0
    assert "extra_tool" not in kinds(ex)
    pat = next(f for f in ex.findings if f.kind == "pattern")
    assert "search_kb x3" in pat.summary
    assert ex.headline.startswith("check max_steps(2) failed")
    assert ex.divergences == [] and ex.alignment == []


def test_suite_orders_flaky_first_and_report_shows_it(tmp_path):
    res = run_suite(_agent(skipped), [
        Case("flaky", "help", [tool_called("escalate")]),
        Case("broken", "help", [tool_called("nope")]),
        Case("fine", "help", [tool_called("lookup_order")]),
    ], trials=6)
    exs = explain_suite(res)
    assert [e.case_id for e in exs] == ["flaky", "broken"]
    assert [e.case_id for e in explain_suite(res, only_flaky=True)] == ["flaky"]
    html = render_html(res)
    assert "Why it fails" in html and "skipped escalate" in html and "passing trial" in html

    path = tmp_path / "run.json"
    res.save(str(path))
    assert main(["explain", str(path), "--case", "flaky"]) == 0
    assert main(["explain", str(path), "--case", "fine"]) == 0


def test_cli_explain_json(tmp_path, capsys):
    res, _ = _explain(skipped, [tool_called("escalate")])
    path = tmp_path / "run.json"
    res.save(str(path))
    assert main(["explain", str(path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data[0]["case_id"] == "c" and data[0]["headline"] == "skipped escalate"
    assert data[0]["findings"][0]["kind"] == "failed_check"
