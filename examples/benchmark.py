"""Benchmark a real model with evalloop: reliability, injection resistance and robustness.

One command, one API key, no extra packages:

    # Windows PowerShell:  $env:GROQ_API_KEY="..."      macOS/Linux:  export GROQ_API_KEY=...
    cd examples
    python benchmark.py                          # Groq, its default model, 10 trials
    python benchmark.py --provider openai --model gpt-4.1-mini --trials 10
    python benchmark.py --provider ollama --model llama3.1   # local, no key

It runs three suites against the support agent in real_agent.py:

  core        the 6 real_suite cases, many trials each (pass rate and pass^k reliability)
  injection   the "return policy" case once per built-in attack (does the agent leak a secret?)
  messy       the order cases rewritten with typos, caps and filler (is it brittle?)

Outputs land in benchmark_results/<provider>-<model>/: one HTML report per suite,
the raw JSON runs, and SUMMARY.md with a table you can paste into a README or post.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evalloop import from_provider, injection_cases, render_html, run_suite, variants  # noqa: E402
from evalloop.adapters import PROVIDERS  # noqa: E402

import real_agent  # noqa: E402
from real_suite import cases as CORE  # noqa: E402


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--provider", default=os.environ.get("EVALLOOP_PROVIDER", "groq"), choices=sorted(PROVIDERS))
    ap.add_argument("--model", default=os.environ.get("EVALLOOP_MODEL"))
    ap.add_argument("--trials", type=int, default=10, help="trials per core case (default 10)")
    ap.add_argument("--workers", type=int, default=2, help="parallel calls; keep low on free tiers")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", default="benchmark_results")
    ap.add_argument("--skip", nargs="*", default=[], choices=["core", "injection", "messy"])
    a = ap.parse_args()

    model = a.model or PROVIDERS[a.provider][2]
    agent = from_provider(a.provider, system=real_agent.SYSTEM, tools=real_agent.TOOLS, impls=real_agent.IMPLS, model=model)
    folder = os.path.join(a.out, re.sub(r"[^A-Za-z0-9._-]+", "-", f"{a.provider}-{model}"))
    os.makedirs(folder, exist_ok=True)
    meta = {"provider": a.provider, "model": model}

    by_id = {c.id: c for c in CORE}
    suites = {
        "core": (CORE, a.trials),
        "injection": (injection_cases(by_id["prompt_injection_in_kb"], "search_kb", real_agent.SECRET), 3),
        "messy": ([v for cid in ("track_order", "refund_small_order", "unknown_order_not_invented") for v in variants(by_id[cid])], 3),
    }

    rows = []
    for name, (cases, trials) in suites.items():
        if name in a.skip:
            continue
        print(f"\n== {name}: {len(cases)} cases x {trials} trials on {a.provider}/{model}")
        t0 = time.time()
        res = run_suite(agent, cases, trials=trials, label=model, suite=f"benchmark-{name}", meta=meta,
                        workers=a.workers, timeout=a.timeout,
                        on_trial=lambda c, tr: print("." if tr.passed else "F", end="", flush=True))
        took = time.time() - t0
        res.save(os.path.join(folder, f"{name}.json"))
        with open(os.path.join(folder, f"{name}.html"), "w", encoding="utf-8") as f:
            f.write(render_html(res))
        lo, hi = res.pass_interval
        errors = sum(1 for c in res.cases for t in c.trials if t.trace.error)
        rows.append((name, len(cases), trials, res.pass_rate, lo, hi, res.reliability,
                     res.latency_p95, res.cost_per_run, errors, took, [c.id for c in res.cases if 0 < c.pass_rate < 1]))
        print(f"\n   pass rate {pct(res.pass_rate)} (95% CI {pct(lo)} to {pct(hi)}), reliability {pct(res.reliability)}, "
              f"p95 {res.latency_p95 / 1000:.1f}s, {took:.0f}s total")

    lines = [
        f"# evalloop benchmark: {a.provider} / {model}",
        "",
        f"Agent: examples/real_agent.py (support agent with 4 tools). Date: {time.strftime('%Y-%m-%d')}.",
        "",
        "| Suite | Cases x trials | Pass rate | 95% CI | Reliability (pass^k) | p95 latency | Cost per run | Errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, n, k, pr, lo, hi, rel, p95, cost, err, _, _ in rows:
        lines.append(f"| {name} | {n} x {k} | {pct(pr)} | {pct(lo)} to {pct(hi)} | {pct(rel)} | {p95 / 1000:.1f}s | ${cost:.4f} | {err} |")
    flaky = [(r[0], r[11]) for r in rows if r[11]]
    if flaky:
        lines += ["", "Flaky cases (passed some trials, failed others):", ""]
        lines += [f"- {suite}: {', '.join(ids)}" for suite, ids in flaky]
    lines += ["", "Cost uses evalloop's default token prices unless you pass your own; check your provider's pricing.",
              "Errors are trials where the API call failed (rate limits, timeouts). Re-run if this is high.", ""]
    with open(os.path.join(folder, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"Reports: {os.path.abspath(folder)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
