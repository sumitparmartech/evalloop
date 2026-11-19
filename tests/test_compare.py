"""compare_runs: several runs of one suite side by side (usually one per model)."""
from evalloop import Case, Tracer, compare_runs, render_compare_html, run_suite, tool_called
from evalloop.cli import main


def model(name, fail_on, cost, ms):
    def agent(text, ctx):
        t = Tracer(text)
        t.llm(name, text, "x", tokens_in=1000, tokens_out=100, latency_ms=ms, price_in=cost, price_out=cost)
        if not (ctx.case.id in fail_on and ctx.trial % 2 == 0):
            t.tool("lookup", {"id": 1}, output={}, latency_ms=5)
        return t.finish("ok")
    return agent


CASES = [Case("a", "q", [tool_called("lookup")]), Case("b", "q", [tool_called("lookup")]), Case("c", "q", [tool_called("lookup")])]


def runs():
    big = run_suite(model("big", set(), 5.0, 900), CASES, trials=4, label="big-model")
    small = run_suite(model("small", {"b"}, 0.5, 300), CASES, trials=4, label="small-model")
    return big, small


def test_table_picks_best_cheapest_fastest():
    big, small = runs()
    cmp = compare_runs([big, small])
    assert [r.label for r in cmp.rows] == ["big-model", "small-model"]
    assert cmp.best("reliability").label == "big-model"
    assert cmp.best("cost").label == "small-model"
    assert cmp.best("speed").label == "small-model"
    small_row = cmp.rows[1]
    assert small_row.case_rates == {"a": 1.0, "b": 0.5, "c": 1.0}
    assert abs(small_row.reliability - 2 / 3) < 1e-9
    txt = cmp.text()
    assert "most reliable: big-model" in txt and "cheapest:      small-model" in txt
    assert cmp.warnings == []


def test_warns_on_missing_cases_and_noise():
    big, small = runs()
    partial = run_suite(model("p", set(), 1.0, 100), CASES[:2], trials=2, label="partial")
    cmp = compare_runs([big, small, partial], labels=["A", "B", "C"])
    assert [r.label for r in cmp.rows] == ["A", "B", "C"]
    assert any("missing" in w for w in cmp.warnings)
    assert "may be noise" in cmp.text()  # 12 trials each: the intervals overlap


def test_html_and_cli(tmp_path):
    big, small = runs()
    html = render_compare_html(compare_runs([big, small]))
    assert "most reliable" in html and "Pass rate per case" in html and "50%" in html
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    big.save(str(a)); small.save(str(b))
    out = tmp_path / "cmp.html"
    assert main(["compare", str(a), str(b), "--labels", "Big,Small", "--html", str(out)]) == 0
    assert "Big" in out.read_text(encoding="utf-8")
