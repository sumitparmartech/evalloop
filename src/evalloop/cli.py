"""Command line interface: run, report, diff."""
from __future__ import annotations

import argparse
import importlib
import os
import sys

from .report import render_html
from .runner import SuiteResult, diff_runs, run_suite


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
    print(f"  reliability   {_fmt(s['reliability'])}  (cases passing every trial)")
    print(f"  latency p95   {s['latency_p95_ms']:.0f} ms")
    print(f"  cost per run  ${s['cost_per_run_usd']:.4f}")
    for c in r.cases:
        mark = "PASS" if c.pass_all else "FAIL"
        print(f"  [{mark}] {c.id:<24} {c.pass_rate * 100:>4.0f}%")


def _print_diff(d, base_label: str, new_label: str) -> None:
    print(f"diff {base_label} -> {new_label}")
    for r in d.regressed:
        print(f"  REGRESSED {r['id']}: {_fmt(r['before'])} to {_fmt(r['after'])}")
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
    r.add_argument("--baseline", help="JSON run to compare against")
    r.add_argument("--fail-under", type=float, help="exit 1 if pass rate is below this (0 to 1)")
    r.add_argument("--fail-on-regression", action="store_true", help="exit 1 if any case got worse than baseline")

    rp = sub.add_parser("report", help="build an HTML report from a saved run")
    rp.add_argument("run")
    rp.add_argument("--baseline")
    rp.add_argument("--html", required=True)

    d = sub.add_parser("diff", help="compare two saved runs")
    d.add_argument("base")
    d.add_argument("new")
    d.add_argument("--fail-on-regression", action="store_true")

    a = p.parse_args(argv)

    if a.cmd == "run":
        cases = _load(a.suite)
        agent = _load(a.agent)
        res = run_suite(agent, cases, trials=a.trials, label=a.label, suite=a.name or a.suite.split(":")[-1], seed=a.seed)
        base = SuiteResult.load(a.baseline) if a.baseline else None
        _print_summary(res)
        code = 0
        if base:
            dif = diff_runs(base, res)
            _print_diff(dif, base.label, res.label)
            if a.fail_on_regression and dif.has_regressions:
                code = 1
        if a.fail_under is not None and res.pass_rate < a.fail_under:
            print(f"FAILED: pass rate {_fmt(res.pass_rate)} is below {_fmt(a.fail_under)}")
            code = 1
        if a.out:
            res.save(a.out)
        if a.html:
            with open(a.html, "w", encoding="utf-8") as f:
                f.write(render_html(res, base))
            print(f"report written to {a.html}")
        return code

    if a.cmd == "report":
        res = SuiteResult.load(a.run)
        base = SuiteResult.load(a.baseline) if a.baseline else None
        with open(a.html, "w", encoding="utf-8") as f:
            f.write(render_html(res, base))
        print(f"report written to {a.html}")
        return 0

    if a.cmd == "diff":
        base, new = SuiteResult.load(a.base), SuiteResult.load(a.new)
        dif = diff_runs(base, new)
        _print_diff(dif, base.label, new.label)
        return 1 if (a.fail_on_regression and dif.has_regressions) else 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
