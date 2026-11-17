"""Command line interface: run, report, diff."""
from __future__ import annotations

import argparse
import importlib
import os
import sys

from .ci import to_junit, to_markdown
from .report import render_html
from .runner import SuiteResult, diff_runs, run_suite
from .store import latest, list_runs, render_trend, save_run, suites_in


def _load(spec: str):
    """Load 'package.module:attribute' relative to the current directory."""
    if ":" not in spec:
        raise SystemExit(f"expected module:attribute, got {spec!r}")
    sys.path.insert(0, os.getcwd())
    mod, attr = spec.split(":", 1)
    return getattr(importlib.import_module(mod), attr)


def _fmt(x: float) -> str:
    return f"{x * 100:.0f}%"


def _print_summary(r: SuiteResult) -> None:
    s = r.summary()
    print(f"suite: {r.suite}   run: {r.label}   ({len(r.cases)} cases x {r.trials_per_case} trials)")
    print(f"  pass rate     {_fmt(s['pass_rate'])}")
    print(f"  pass rate CI  {_fmt(s['pass_rate_low'])} to {_fmt(s['pass_rate_high'])}  (95%)")
    print(f"  reliability   {_fmt(s['reliability'])}  (cases passing every trial)")
    if r.flaky_cases:
        print(f"  flaky cases   {', '.join(r.flaky_cases)}")
    print(f"  latency p95   {s['latency_p95_ms']:.0f} ms")
    print(f"  cost per run  ${s['cost_per_run_usd']:.4f}")
    for c in r.cases:
        mark = "PASS" if c.pass_all else "FAIL"
        print(f"  [{mark}] {c.id:<24} {c.pass_rate * 100:>4.0f}%")


def _print_diff(d, base_label: str, new_label: str) -> None:
    print(f"diff {base_label} -> {new_label}")
    for r in d.regressed:
        tag = "  (may be noise, p=%.2f)" % r["p_value"] if r["noise"] else ""
        print(f"  REGRESSED {r['id']}: {_fmt(r['before'])} to {_fmt(r['after'])}{tag}")
    for r in d.improved:
        print(f"  improved  {r['id']}: {_fmt(r['before'])} to {_fmt(r['after'])}")
    print(f"  {len(d.unchanged)} unchanged, {len(d.added)} added, {len(d.removed)} removed")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="evalloop", description="Unit tests and a CI gate for LLM agents.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a suite against an agent")
    r.add_argument("suite", help="module:attribute that holds a list of Case")
    r.add_argument("--agent", required=True, help="module:attribute of the agent function")
    r.add_argument("--trials", type=int, default=5)
    r.add_argument("--label", default="run")
    r.add_argument("--name", help="suite name shown in the report")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", help="save the run as JSON")
    r.add_argument("--html", help="write an HTML report")
    r.add_argument("--baseline", help="JSON run to compare against, or 'last' to use the newest run in --store")
    r.add_argument("--store", help="folder that keeps every run, for history and trends")
    r.add_argument("--workers", type=int, default=1, help="run trials in parallel threads")
    r.add_argument("--timeout", type=float, help="seconds allowed per trial")
    r.add_argument("--alpha", type=float, help="count a drop as a regression only if p < alpha (ignore noise)")
    r.add_argument("--junit", help="write JUnit XML")
    r.add_argument("--markdown", help="write a Markdown summary (use $GITHUB_STEP_SUMMARY on GitHub)")
    r.add_argument("--fail-under", type=float, help="exit 1 if pass rate is below this (0 to 1)")
    r.add_argument("--fail-on-regression", action="store_true", help="exit 1 if any case got worse than baseline")

    tr = sub.add_parser("trend", help="build an HTML trend report from a run store")
    tr.add_argument("store")
    tr.add_argument("--suite", help="suite name (optional if the store has one suite)")
    tr.add_argument("--html", required=True)

    ini = sub.add_parser("init", help="scaffold an evals/ folder with an example agent, suite and CI workflow")
    ini.add_argument("--dir", default="evals")

    rp = sub.add_parser("report", help="build an HTML report from a saved run")
    rp.add_argument("run")
    rp.add_argument("--baseline")
    rp.add_argument("--html", required=True)

    d = sub.add_parser("diff", help="compare two saved runs")
    d.add_argument("base")
    d.add_argument("new")
    d.add_argument("--fail-on-regression", action="store_true")
    d.add_argument("--alpha", type=float, help="ignore drops that are likely noise (p >= alpha)")

    a = p.parse_args(argv)

    if a.cmd == "run":
        cases = _load(a.suite)
        agent = _load(a.agent)
        res = run_suite(agent, cases, trials=a.trials, label=a.label, suite=a.name or a.suite.split(":")[-1], seed=a.seed, workers=a.workers, timeout=a.timeout)
        if a.baseline == "last":
            if not a.store:
                raise SystemExit("--baseline last needs --store")
            base = latest(a.store, res.suite)
            if base is None:
                print("no earlier run in the store: this run becomes the first baseline")
        else:
            base = SuiteResult.load(a.baseline) if a.baseline else None
        _print_summary(res)
        code = 0
        dif = None
        if base:
            dif = diff_runs(base, res, alpha=a.alpha)
            _print_diff(dif, base.label, res.label)
            if a.fail_on_regression and dif.has_regressions:
                code = 1
        if a.fail_under is not None and res.pass_rate < a.fail_under:
            print(f"FAILED: pass rate {_fmt(res.pass_rate)} is below {_fmt(a.fail_under)}")
            code = 1
        if a.out:
            res.save(a.out)
        if a.store:
            print(f"saved to {save_run(a.store, res)}")
        if a.junit:
            with open(a.junit, "w", encoding="utf-8") as f:
                f.write(to_junit(res))
        if a.markdown:
            with open(a.markdown, "a", encoding="utf-8") as f:
                f.write(to_markdown(res, dif if base else None, base))
        if a.html:
            with open(a.html, "w", encoding="utf-8") as f:
                f.write(render_html(res, base))
            print(f"report written to {a.html}")
        return code

    if a.cmd == "trend":
        suite = a.suite
        if not suite:
            found = suites_in(a.store)
            if len(found) != 1:
                raise SystemExit(f"pick one with --suite; the store has: {', '.join(found) or 'nothing'}")
            suite = found[0]
        runs = list_runs(a.store, suite)
        with open(a.html, "w", encoding="utf-8") as f:
            f.write(render_trend(runs))
        print(f"trend report for {len(runs)} runs written to {a.html}")
        return 0

    if a.cmd == "init":
        from .scaffold import scaffold

        for path in scaffold(a.dir):
            print("created", path)
        print(f"\nNext:  cd {a.dir} && evalloop run suite:cases --agent agent:support_agent --trials 5 --store runs --baseline last")
        return 0

    if a.cmd == "report":
        res = SuiteResult.load(a.run)
        base = SuiteResult.load(a.baseline) if a.baseline else None
        with open(a.html, "w", encoding="utf-8") as f:
            f.write(render_html(res, base))
        print(f"report written to {a.html}")
        return 0

    if a.cmd == "diff":
        base, new = SuiteResult.load(a.base), SuiteResult.load(a.new)
        dif = diff_runs(base, new, alpha=a.alpha)
        _print_diff(dif, base.label, new.label)
        return 1 if (a.fail_on_regression and dif.has_regressions) else 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
