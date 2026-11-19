"""Compare saved runs of the same suite side by side, usually one run per model.

    evalloop compare runs/gpt-4.1-mini.json runs/llama-70b.json runs/gpt-oss-120b.json --html compare.html

The table answers the question teams actually ask before switching models: which one is
reliable enough, and what does it cost and how fast is it? Every number comes from the runs;
nothing is estimated. Pass prices to `run_suite` (or `--price-in/--price-out` in the benchmark
script) if you want the cost column to mean something.
"""
from __future__ import annotations

import dataclasses as dc
import html
from typing import Optional

from .runner import SuiteResult

__all__ = ["ModelRow", "Comparison", "compare_runs", "render_compare_html"]


@dc.dataclass
class ModelRow:
    label: str
    pass_rate: float
    interval: tuple
    reliability: float
    p95_ms: float
    cost_per_run: float
    trials: int
    errors: int
    case_rates: dict


@dc.dataclass
class Comparison:
    suite: str
    rows: list
    case_ids: list
    warnings: list

    def best(self, key: str) -> Optional[ModelRow]:
        if not self.rows:
            return None
        if key == "reliability":
            return max(self.rows, key=lambda r: (r.reliability, r.pass_rate, -r.cost_per_run))
        if key == "cost":
            priced = [r for r in self.rows if r.cost_per_run > 0]
            return min(priced, key=lambda r: r.cost_per_run) if priced else None
        if key == "speed":
            return min(self.rows, key=lambda r: r.p95_ms)
        raise ValueError(key)

    def text(self) -> str:
        w = max([len(r.label) for r in self.rows] + [5])
        head = f"{'model':<{w}}  {'pass rate':>9}  {'95% CI':>13}  {'reliable':>8}  {'p95':>7}  {'cost/run':>9}  {'errors':>6}"
        lines = [f"suite: {self.suite}", head, "-" * len(head)]
        for r in self.rows:
            ci = f"{r.interval[0] * 100:.0f}-{r.interval[1] * 100:.0f}%"
            cost = f"${r.cost_per_run:.4f}" if r.cost_per_run else "not set"
            lines.append(f"{r.label:<{w}}  {r.pass_rate * 100:>8.0f}%  {ci:>13}  {r.reliability * 100:>7.0f}%  {r.p95_ms / 1000:>6.1f}s  {cost:>9}  {r.errors:>6}")
        b = self.best("reliability")
        if b:
            lines.append("")
            lines.append(f"most reliable: {b.label} ({b.reliability * 100:.0f}% of cases passed every trial)")
        c = self.best("cost")
        if c:
            lines.append(f"cheapest:      {c.label} (${c.cost_per_run:.4f} per run)")
        s = self.best("speed")
        if s:
            lines.append(f"fastest p95:   {s.label} ({s.p95_ms / 1000:.1f}s)")
        if len(self.rows) > 1 and self._overlap():
            lines.append("note: the pass-rate intervals overlap, so the difference may be noise. Run more trials to separate them.")
        lines += [f"warning: {x}" for x in self.warnings]
        return "\n".join(lines)

    def _overlap(self) -> bool:
        rs = sorted(self.rows, key=lambda r: -r.pass_rate)
        return rs[0].interval[0] <= rs[1].interval[1]


def _errors(res: SuiteResult) -> int:
    return sum(1 for c in res.cases for t in c.trials if t.trace.error)


def _label(res: SuiteResult) -> str:
    m = res.meta.get("model") if isinstance(res.meta, dict) else None
    if m and res.label in ("run", m):
        prov = res.meta.get("provider")
        return f"{prov}/{m}" if prov else m
    return res.label


def compare_runs(runs: list, labels: Optional[list] = None) -> Comparison:
    """Build a comparison from SuiteResult objects (one per model or prompt version)."""
    if not runs:
        raise ValueError("compare_runs needs at least one run")
    warnings = []
    suites = {r.suite for r in runs}
    if len(suites) > 1:
        warnings.append(f"runs come from different suites ({', '.join(sorted(suites))}); compare with care")
    ids: list = []
    for r in runs:
        for c in r.cases:
            if c.id not in ids:
                ids.append(c.id)
    common = set(ids)
    for r in runs:
        common &= {c.id for c in r.cases}
    if len(common) < len(ids):
        warnings.append(f"{len(ids) - len(common)} case(s) are missing from some runs")
    rows = []
    for i, r in enumerate(runs):
        name = labels[i] if labels and i < len(labels) else _label(r)
        errs = _errors(r)
        if errs:
            warnings.append(f"{name}: {errs} trial(s) failed with an API error, which counts as a failure")
        rows.append(ModelRow(name, r.pass_rate, r.pass_interval, r.reliability, r.latency_p95, r.cost_per_run,
                             sum(len(c.trials) for c in r.cases), errs, {c.id: c.pass_rate for c in r.cases}))
    return Comparison(next(iter(suites)) if len(suites) == 1 else "mixed", rows, ids, warnings)


def _e(x) -> str:
    return html.escape(str(x))


def render_compare_html(cmp: Comparison) -> str:
    """One self-contained HTML page: summary table plus a per-case matrix."""
    best = cmp.best("reliability")
    cheap = cmp.best("cost")
    fast = cmp.best("speed")
    maxp95 = max((r.p95_ms for r in cmp.rows), default=1) or 1
    maxcost = max((r.cost_per_run for r in cmp.rows), default=0) or 1

    def badge(r):
        out = []
        if best is r:
            out.append('<span class="b g">most reliable</span>')
        if cheap is r:
            out.append('<span class="b a">cheapest</span>')
        if fast is r:
            out.append('<span class="b a">fastest</span>')
        return " ".join(out)

    def cost_html(r):
        if not r.cost_per_run:
            return '<span class="mu">not set</span>'
        return f'<div class="bar s"><i style="width:{r.cost_per_run / maxcost * 100:.1f}%"></i></div>${r.cost_per_run:.4f}'

    def row(r):
        return (
            f'<tr><td class="m">{_e(r.label)} {badge(r)}</td>'
            f'<td><div class="bar"><i style="width:{r.pass_rate * 100:.1f}%"></i></div><b>{r.pass_rate * 100:.0f}%</b> '
            f'<span class="mu">{r.interval[0] * 100:.0f} to {r.interval[1] * 100:.0f}%</span></td>'
            f'<td><b>{r.reliability * 100:.0f}%</b></td>'
            f'<td><div class="bar s"><i style="width:{r.p95_ms / maxp95 * 100:.1f}%"></i></div>{r.p95_ms / 1000:.1f}s</td>'
            f'<td>{cost_html(r)}</td><td>{r.errors}</td></tr>'
        )

    rows = "".join(row(r) for r in cmp.rows)

    def cell(v):
        if v is None:
            return '<td class="c na">-</td>'
        cls = "ok" if v == 1 else ("bad" if v == 0 else "mid")
        return f'<td class="c {cls}">{v * 100:.0f}%</td>'

    matrix = "".join(
        f'<tr><td class="m">{_e(cid)}</td>{"".join(cell(r.case_rates.get(cid)) for r in cmp.rows)}</tr>' for cid in cmp.case_ids
    )
    heads = "".join(f"<th>{_e(r.label)}</th>" for r in cmp.rows)
    warn = "".join(f'<div class="w">{_e(w)}</div>' for w in cmp.warnings)
    note = ""
    if len(cmp.rows) > 1 and cmp._overlap():
        note = '<div class="w">The top two pass-rate intervals overlap, so the gap may be noise. Run more trials to separate them.</div>'
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>evalloop compare: {_e(cmp.suite)}</title><style>
:root{{--bg:#F5F0E6;--card:#FFFDF8;--ink:#1B1A17;--mute:#5B564B;--line:#E4DBC9;--ok:#0B7A4F;--okbg:#DCF4E9;--bad:#B3261E;--badbg:#FBE1DE;--acc:#9A4A00;--amber:#F5A524;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14120E;--card:#1D1A14;--ink:#F3EEE3;--mute:#A8A08F;--line:#332E24;--ok:#46D39A;--okbg:#10291f;--bad:#FF8A80;--badbg:#321615;--acc:#F5A524}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:1040px;margin:0 auto;padding:32px 20px 64px}}.tag{{font:600 12px var(--mono);letter-spacing:.06em;color:var(--acc);text-transform:uppercase}}
h1{{margin:4px 0 18px;font-size:28px;letter-spacing:-.02em}}h2{{font-size:13px;color:var(--mute);text-transform:uppercase;letter-spacing:.06em;margin:30px 0 10px}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;font-size:14px}}
th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}}th{{color:var(--mute);font-size:12px;font-weight:600}}
.m{{font:600 13px var(--mono)}}.mu{{color:var(--mute);font:12px var(--mono)}}
.bar{{height:8px;width:140px;border-radius:99px;background:var(--line);overflow:hidden;display:inline-block;vertical-align:middle;margin-right:8px}}
.bar i{{display:block;height:100%;background:var(--ok);border-radius:99px}}.bar.s{{width:80px}}.bar.s i{{background:var(--amber)}}
.b{{display:inline-block;font:600 11px var(--mono);padding:2px 7px;border-radius:99px;margin-left:6px}}.b.g{{background:var(--okbg);color:var(--ok)}}.b.a{{background:var(--line);color:var(--mute)}}
td.c{{text-align:center;font:600 13px var(--mono)}}td.ok{{color:var(--ok);background:var(--okbg)}}td.bad{{color:var(--bad);background:var(--badbg)}}td.mid{{color:var(--acc)}}td.na{{color:var(--mute)}}
.w{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--amber);border-radius:10px;padding:10px 14px;margin:12px 0;font-size:14px}}
footer{{margin-top:36px;color:var(--mute);font-size:12px}}
</style></head><body><div class="wrap">
<div class="tag">evalloop compare</div><h1>{_e(cmp.suite)}: {len(cmp.rows)} runs side by side</h1>
{warn}{note}
<table><tr><th>Model / run</th><th>Pass rate (95% CI)</th><th>Reliable (pass^k)</th><th>p95 latency</th><th>Cost per run</th><th>API errors</th></tr>{rows}</table>
<h2>Pass rate per case</h2>
<table><tr><th>Case</th>{heads}</tr>{matrix}</table>
<footer>Reliable means the share of cases that passed every one of their trials. Generated by evalloop.</footer>
</div></body></html>"""
