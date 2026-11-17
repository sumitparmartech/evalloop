"""A folder of past runs, so you can compare with the last run and watch trends over time.

Layout:  <store>/<suite>/<UTC time>_<label>.json   (plain JSON, safe to commit or cache in CI)
"""
from __future__ import annotations

import html
import os
import re
from typing import List, Optional

from .runner import SuiteResult


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "run"


def save_run(store: str, result: SuiteResult) -> str:
    folder = os.path.join(store, _slug(result.suite))
    os.makedirs(folder, exist_ok=True)
    stamp = re.sub(r"[^0-9T]", "", result.created.split("+")[0])
    path = os.path.join(folder, f"{stamp}_{_slug(result.label)}.json")
    n = 1
    while os.path.exists(path):  # two runs in the same second
        n += 1
        path = os.path.join(folder, f"{stamp}_{_slug(result.label)}-{n}.json")
    result.save(path)
    return path


def list_runs(store: str, suite: str) -> List[SuiteResult]:
    folder = os.path.join(store, _slug(suite))
    if not os.path.isdir(folder):
        return []
    runs = [SuiteResult.load(os.path.join(folder, f)) for f in sorted(os.listdir(folder)) if f.endswith(".json")]
    return sorted(runs, key=lambda r: r.created)


def latest(store: str, suite: str) -> Optional[SuiteResult]:
    runs = list_runs(store, suite)
    return runs[-1] if runs else None


def suites_in(store: str) -> List[str]:
    if not os.path.isdir(store):
        return []
    return sorted(d for d in os.listdir(store) if os.path.isdir(os.path.join(store, d)))


# ---------------------------------------------------------------- trend report

def _chart(title: str, values: list, labels: list, fmt, lo=None, hi=None, band=None, color="#2563eb") -> str:
    w, h, pl, pr, pt, pb = 520, 170, 46, 14, 26, 26
    if not values:
        return ""
    vmin = min(values) if lo is None else lo
    vmax = max(values) if hi is None else hi
    if band:
        vmin, vmax = (min(vmin, min(b[0] for b in band)), max(vmax, max(b[1] for b in band))) if lo is None else (vmin, vmax)
    if vmax - vmin < 1e-12:
        vmax, vmin = vmax + 1, vmin - 1 if vmin else vmin
        vmin = max(vmin, 0) if min(values) >= 0 else vmin
    n = len(values)
    x = lambda i: pl + (w - pl - pr) * (i / (n - 1) if n > 1 else 0.5)
    y = lambda v: pt + (h - pt - pb) * (1 - (v - vmin) / (vmax - vmin))
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    area = ""
    if band and n > 1:
        top = " ".join(f"{x(i):.1f},{y(b[1]):.1f}" for i, b in enumerate(band))
        bot = " ".join(f"{x(i):.1f},{y(b[0]):.1f}" for i, b in reversed(list(enumerate(band))))
        area = f'<polygon points="{top} {bot}" fill="{color}" opacity="0.13"/>'
    dots = "".join(
        f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3.5" fill="{color}"><title>{html.escape(labels[i])}: {html.escape(fmt(v))}</title></circle>'
        for i, v in enumerate(values)
    )
    grid = "".join(
        f'<line x1="{pl}" x2="{w - pr}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="#e5e7eb"/>'
        f'<text x="{pl - 6}" y="{y(g) + 4:.1f}" text-anchor="end" font-size="10" fill="#6b7280">{html.escape(fmt(g))}</text>'
        for g in (vmin, (vmin + vmax) / 2, vmax)
    )
    return (
        f'<div class="chart"><h3>{html.escape(title)}</h3>'
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">{grid}{area}'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>{dots}</svg></div>'
    )


def render_trend(runs: List[SuiteResult]) -> str:
    if not runs:
        return "<p>No runs yet.</p>"
    pct = lambda v: f"{v * 100:.0f}%"
    labels = [f"{r.label} ({r.created[:16].replace('T', ' ')})" for r in runs]
    sums = [r.summary() for r in runs]
    charts = "".join(
        [
            _chart("Pass rate (shaded: 95% confidence interval)", [s["pass_rate"] for s in sums], labels, pct, 0, 1,
                   band=[(s["pass_rate_low"], s["pass_rate_high"]) for s in sums]),
            _chart("Reliability (pass^k)", [s["reliability"] for s in sums], labels, pct, 0, 1, color="#059669"),
            _chart("Latency p95", [s["latency_p95_ms"] for s in sums], labels, lambda v: f"{v:.0f} ms", color="#d97706"),
            _chart("Cost per run", [s["cost_per_run_usd"] for s in sums], labels, lambda v: f"${v:.4f}", color="#7c3aed"),
            _chart("Flaky cases", [float(s["flaky_cases"]) for s in sums], labels, lambda v: f"{v:.0f}", color="#dc2626"),
        ]
    )
    rows = "".join(
        f"<tr><td>{html.escape(r.label)}</td><td>{html.escape(r.created[:19].replace('T', ' '))}</td>"
        f"<td>{pct(s['pass_rate'])}</td><td>{pct(s['reliability'])}</td><td>{s['latency_p95_ms']:.0f} ms</td>"
        f"<td>${s['cost_per_run_usd']:.4f}</td><td>{s['flaky_cases']}</td></tr>"
        for r, s in zip(reversed(runs), reversed(sums))
    )
    css = (
        "body{font:15px/1.5 system-ui,sans-serif;background:#f6f7f9;color:#111827;margin:0}.wrap{max-width:1100px;margin:0 auto;padding:36px 24px}"
        "h1{margin:4px 0 2px;font-size:28px}.tag{font:600 12px ui-monospace,monospace;color:#2563eb;letter-spacing:.08em}"
        ".sub{color:#6b7280;margin-bottom:22px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}"
        ".chart{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:14px 16px}.chart h3{margin:0 0 4px;font-size:14px;color:#374151}"
        "svg{width:100%;height:auto}table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e5e7eb;border-radius:14px;margin-top:22px}"
        "th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #f0f1f3;font-size:14px}th{color:#6b7280;font-weight:600}"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>evalloop trend: {html.escape(runs[-1].suite)}</title><style>{css}</style></head><body><div class=\"wrap\">"
        f'<div class="tag">evalloop trend</div><h1>{html.escape(runs[-1].suite)}</h1>'
        f'<div class="sub">{len(runs)} runs, from {html.escape(runs[0].created[:10])} to {html.escape(runs[-1].created[:10])}</div>'
        f'<div class="grid">{charts}</div>'
        f"<table><tr><th>Run</th><th>When (UTC)</th><th>Pass rate</th><th>Reliability</th><th>p95</th><th>Cost/run</th><th>Flaky</th></tr>{rows}</table>"
        "</div></body></html>"
    )
