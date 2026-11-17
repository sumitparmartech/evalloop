"""Outputs for CI systems and test runners: JUnit XML, a Markdown PR summary, and a pytest helper."""
from __future__ import annotations

from typing import Callable, Optional
from xml.sax.saxutils import escape, quoteattr

from .runner import Case, Diff, SuiteResult, run_suite


def to_junit(result: SuiteResult) -> str:
    """JUnit XML. Every case is a test, so Jenkins, GitLab, CircleCI and GitHub show it natively."""
    fails = sum(1 for c in result.cases if not c.pass_all)
    out = ['<?xml version="1.0" encoding="UTF-8"?>']
    out.append(
        f'<testsuite name={quoteattr(result.suite)} tests="{len(result.cases)}" failures="{fails}" '
        f'timestamp={quoteattr(result.created)}>'
    )
    for c in result.cases:
        time_s = sum(t.trace.latency_ms for t in c.trials) / 1000
        out.append(f'  <testcase classname={quoteattr(result.suite)} name={quoteattr(c.id)} time="{time_s:.3f}">')
        if not c.pass_all:
            bad = [
                f"trial {t.trial}: " + "; ".join(f"{k.name} ({k.detail})" for k in t.checks if not k.passed)
                for t in c.trials
                if not t.passed
            ]
            kind = "flaky" if c.flaky else "failed"
            out.append(
                f'    <failure message={quoteattr(f"{kind}: passed {c.passes}/{len(c.trials)} trials")}>'
                f"{escape(chr(10).join(bad))}</failure>"
            )
        out.append("  </testcase>")
    out.append("</testsuite>")
    return "\n".join(out) + "\n"


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def to_markdown(result: SuiteResult, diff: Optional[Diff] = None, base: Optional[SuiteResult] = None) -> str:
    """A short report for a pull request comment or the GitHub job summary."""
    s = result.summary()
    lo, hi = s["pass_rate_low"], s["pass_rate_high"]
    lines = [f"### evalloop: {result.suite} ({result.label})", ""]
    lines.append(
        f"**Pass rate {_pct(s['pass_rate'])}** (95% CI {_pct(lo)} to {_pct(hi)}) | "
        f"**Reliability {_pct(s['reliability'])}** | p95 latency {s['latency_p95_ms']:.0f} ms | "
        f"cost/run ${s['cost_per_run_usd']:.4f}"
    )
    if result.flaky_cases:
        lines += ["", f"Flaky cases (pass some trials, fail others): {', '.join(result.flaky_cases)}"]
    if diff is not None:
        lines += ["", f"**vs {base.label if base else 'baseline'}**"]
        if diff.regressed:
            for r in diff.regressed:
                note = " (could be noise)" if r["noise"] else ""
                lines.append(f"- REGRESSED `{r['id']}`: {_pct(r['before'])} to {_pct(r['after'])}{note}")
        if diff.improved:
            for r in diff.improved:
                lines.append(f"- improved `{r['id']}`: {_pct(r['before'])} to {_pct(r['after'])}")
        if not diff.regressed and not diff.improved:
            lines.append("- no change")
    failing = [c for c in result.cases if not c.pass_all]
    if failing:
        lines += ["", "| Case | Passed | Why it failed |", "|---|---|---|"]
        for c in failing:
            t = next(t for t in c.trials if not t.passed)
            why = "; ".join(f"{k.name}: {k.detail}" for k in t.checks if not k.passed)[:160].replace("|", "/")
            lines.append(f"| `{c.id}` | {c.passes}/{len(c.trials)} | {why} |")
    return "\n".join(lines) + "\n"


def assert_reliable(
    agent: Callable,
    case: Case,
    trials: int = 5,
    min_pass_rate: float = 1.0,
    workers: int = 1,
) -> None:
    """Use inside pytest: fails the test with a readable reason if the agent is not reliable enough.

        def test_refund_flow():
            assert_reliable(my_agent, refund_case, trials=5, min_pass_rate=0.8)
    """
    res = run_suite(agent, [case], trials=trials, workers=workers)
    c = res.cases[0]
    if c.pass_rate + 1e-9 < min_pass_rate:
        reasons = []
        for t in c.trials:
            if not t.passed:
                reasons.append(f"  trial {t.trial}: " + "; ".join(f"{k.name} ({k.detail})" for k in t.checks if not k.passed))
        raise AssertionError(
            f"case {case.id!r} passed {c.passes}/{trials} trials (needed {min_pass_rate:.0%}):\n" + "\n".join(reasons)
        )
