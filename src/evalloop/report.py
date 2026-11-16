"""Single-file HTML report. No JavaScript, no external assets, safe to attach to a PR."""
from __future__ import annotations

import html
from typing import Optional

from .core import text_of
from .runner import CaseResult, SuiteResult, TrialResult, diff_runs

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#14171f;--mute:#5c6573;--line:#e3e6ec;--ok:#0f7b4d;--okbg:#e4f5ec;--bad:#b42318;--badbg:#fdeceb;--acc:#2f5bea;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#0e1116;--card:#161b22;--ink:#e8ebf0;--mute:#9aa4b2;--line:#262d38;--ok:#4cc38a;--okbg:#10291f;--bad:#ff7b72;--badbg:#321615;--acc:#7b9cff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:32px 20px 64px}
.tag{font:600 12px var(--mono);letter-spacing:.04em;color:var(--acc);text-transform:uppercase}
h1{margin:4px 0 4px;font-size:28px;letter-spacing:-.01em}.sub{color:var(--mute);font-size:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:24px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.kpi .l{color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.kpi .v{font-size:30px;font-weight:650;letter-spacing:-.02em;margin-top:2px}
.kpi .n{color:var(--mute);font-size:12px;margin-top:2px}
.d{display:inline-block;font:600 12px var(--mono);padding:2px 7px;border-radius:99px;margin-left:8px;vertical-align:middle}
.d.good{color:var(--ok);background:var(--okbg)}.d.badd{color:var(--bad);background:var(--badbg)}.d.flat{color:var(--mute);background:var(--line)}
.banner{border:1px solid var(--line);background:var(--card);border-radius:12px;padding:14px 16px;margin-bottom:20px}
.banner b{font-weight:650}.chip{display:inline-block;font:12px var(--mono);padding:2px 8px;border-radius:99px;margin:4px 6px 0 0}
.chip.bad{background:var(--badbg);color:var(--bad)}.chip.ok{background:var(--okbg);color:var(--ok)}
h2{font-size:15px;margin:28px 0 10px;color:var(--mute);text-transform:uppercase;letter-spacing:.06em;font-weight:600}
details.case{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:8px;overflow:hidden}
details.case[open]{border-color:var(--acc)}
summary{list-style:none;cursor:pointer;display:grid;grid-template-columns:28px 1fr auto auto;gap:12px;align-items:center;padding:12px 14px}
summary::-webkit-details-marker{display:none}
.st{width:24px;height:24px;border-radius:50%;display:grid;place-items:center;font-weight:700;font-size:13px}
.st.ok{background:var(--okbg);color:var(--ok)}.st.bad{background:var(--badbg);color:var(--bad)}
.cid{font:600 14px var(--mono)}.inp{color:var(--mute);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:520px}
.dots{display:flex;gap:3px}.dot{width:10px;height:18px;border-radius:3px}.dot.ok{background:var(--ok)}.dot.bad{background:var(--bad)}
.rate{font:600 14px var(--mono);min-width:92px;text-align:right}
.body{border-top:1px solid var(--line);padding:14px 16px 18px}
.trial{margin-bottom:18px}.trial h3{font-size:13px;margin:0 0 8px;font-weight:650}
.checks{list-style:none;margin:0 0 12px;padding:0}.checks li{font:13px var(--mono);padding:3px 0}
.checks .p{color:var(--ok)}.checks .f{color:var(--bad)}.checks .det{color:var(--mute);font-family:inherit}
.ans{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:13px;margin-bottom:10px}
.tl{border-left:2px solid var(--line);margin-left:6px;padding-left:14px}
.step{position:relative;padding:5px 0;font-size:13px}.step::before{content:"";position:absolute;left:-20px;top:12px;width:10px;height:10px;border-radius:50%;background:var(--card);border:2px solid var(--mute)}
.step.tool::before{border-color:var(--acc)}.step .k{font:600 11px var(--mono);color:var(--mute);text-transform:uppercase;margin-right:6px}
.step.tool .k{color:var(--acc)}.step .m{color:var(--mute);font:12px var(--mono)}.step code{font:12px var(--mono);color:var(--mute);word-break:break-all}
.tbl{width:100%;border-collapse:collapse;font-size:13px;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.tbl th,.tbl td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line)}.tbl th{color:var(--mute);font-weight:600;font-size:12px}
footer{margin-top:40px;color:var(--mute);font-size:12px}

.kpi{animation:rise .6s cubic-bezier(.2,.8,.2,1) both}
.kpi:nth-child(2){animation-delay:.07s}.kpi:nth-child(3){animation-delay:.14s}.kpi:nth-child(4){animation-delay:.21s}.kpi:nth-child(5){animation-delay:.28s}
.banner{animation:rise .6s .2s cubic-bezier(.2,.8,.2,1) both}
.case{animation:rise .55s cubic-bezier(.2,.8,.2,1) both}
.case:nth-of-type(1){animation-delay:.25s}.case:nth-of-type(2){animation-delay:.32s}.case:nth-of-type(3){animation-delay:.39s}.case:nth-of-type(4){animation-delay:.46s}.case:nth-of-type(5){animation-delay:.53s}.case:nth-of-type(n+6){animation-delay:.6s}
.dot{animation:pop .5s cubic-bezier(.2,.9,.2,1.5) both}
.dot:nth-child(2){animation-delay:.05s}.dot:nth-child(3){animation-delay:.1s}.dot:nth-child(4){animation-delay:.15s}.dot:nth-child(5){animation-delay:.2s}.dot:nth-child(n+6){animation-delay:.25s}
.case summary{transition:background .2s}.case summary:hover{background:color-mix(in srgb,var(--line) 35%,transparent)}
@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
@keyframes pop{0%{transform:scale(.2)}60%{transform:scale(1.25)}100%{transform:scale(1)}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _trunc(x, n=260) -> str:
    s = text_of(x)
    return s if len(s) <= n else s[: n - 1] + "..."


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _ms(x: float) -> str:
    return f"{x / 1000:.2f}s" if x >= 1000 else f"{x:.0f}ms"


def _usd(x: float) -> str:
    return f"${x:.4f}" if x < 1 else f"${x:.2f}"


def _delta(value: float, fmt, higher_is_better: bool, tol: float = 1e-9) -> str:
    if abs(value) <= tol:
        return '<span class="d flat">no change</span>'
    good = (value > 0) == higher_is_better
    sign = "+" if value > 0 else "-"
    return f'<span class="d {"good" if good else "badd"}">{sign}{fmt(abs(value))}</span>'


def _kpi(label, value, note, delta_html="") -> str:
    return (
        f'<div class="kpi"><div class="l">{_e(label)}</div>'
        f'<div class="v">{_e(value)}{delta_html}</div><div class="n">{_e(note)}</div></div>'
    )


def _trial_block(t: TrialResult) -> str:
    rows = []
    for c in t.checks:
        mark, cls = ("PASS", "p") if c.passed else ("FAIL", "f")
        det = f' <span class="det">({_e(c.detail)})</span>' if c.detail and not c.passed else ""
        rows.append(f'<li><span class="{cls}">{mark}</span> {_e(c.name)}{det}</li>')
    steps = []
    for s in t.trace.steps:
        meta = f"{_ms(s.latency_ms)}"
        if s.kind == "llm":
            meta += f" / {s.tokens_in}+{s.tokens_out} tok / {_usd(s.cost)}"
        err = f' <span class="m" style="color:var(--bad)">error: {_e(s.error)}</span>' if s.error else ""
        steps.append(
            f'<div class="step {_e(s.kind)}"><span class="k">{_e(s.kind)}</span><b>{_e(s.name)}</b> '
            f'<span class="m">{_e(meta)}</span>{err}<br>'
            f'<code>in: {_e(_trunc(s.input, 160))}</code><br><code>out: {_e(_trunc(s.output, 200))}</code></div>'
        )
    verdict = "passed" if t.passed else "failed"
    return (
        f'<div class="trial"><h3>Trial {t.trial + 1} {verdict}</h3>'
        f'<ul class="checks">{"".join(rows)}</ul>'
        f'<div class="ans"><b>Answer:</b> {_e(_trunc(t.trace.answer, 500)) or "(empty)"}</div>'
        f'<div class="tl">{"".join(steps)}</div></div>'
    )


def _case_block(c: CaseResult, base: Optional[CaseResult]) -> str:
    dots = "".join(f'<span class="dot {"ok" if t.passed else "bad"}" title="trial {t.trial + 1}"></span>' for t in c.trials)
    rate = f"{_pct(c.pass_rate)}"
    delta = ""
    if base is not None:
        delta = _delta(c.pass_rate - base.pass_rate, lambda v: f"{v * 100:.0f}pt", True)
    status = ("ok", "OK") if c.pass_all else ("bad", "!")
    shown = []
    first_bad = next((t for t in c.trials if not t.passed), None)
    first_ok = next((t for t in c.trials if t.passed), None)
    for t in (first_bad, first_ok):
        if t is not None and t not in shown:
            shown.append(t)
    tags = " ".join(f'<span class="chip ok" style="background:var(--line);color:var(--mute)">{_e(t)}</span>' for t in c.tags)
    return (
        f'<details class="case"{"" if c.pass_all else " open"}><summary>'
        f'<span class="st {status[0]}">{status[1] if status[0] == "bad" else "&#10003;"}</span>'
        f'<span><span class="cid">{_e(c.id)}</span> {tags}<div class="inp">{_e(c.input)}</div></span>'
        f'<span class="dots">{dots}</span><span class="rate">{rate}{delta}</span></summary>'
        f'<div class="body">{"".join(_trial_block(t) for t in shown)}</div></details>'
    )


def render_html(result: SuiteResult, baseline: Optional[SuiteResult] = None) -> str:
    s = result.summary()
    bs = baseline.summary() if baseline else None
    d = diff_runs(baseline, result) if baseline else None

    def dl(key, fmt, hib):
        return _delta(s[key] - bs[key], fmt, hib) if bs else ""

    k = result.trials_per_case
    kpis = "".join(
        [
            _kpi("Pass rate", _pct(s["pass_rate"]), f"{k} trials per case", dl("pass_rate", lambda v: f"{v * 100:.0f}pt", True)),
            _kpi("Reliability (pass^k)", _pct(s["reliability"]), f"cases that passed all {k} trials", dl("reliability", lambda v: f"{v * 100:.0f}pt", True)),
            _kpi("Latency p95", _ms(s["latency_p95_ms"]), f"p50 {_ms(s['latency_p50_ms'])}", dl("latency_p95_ms", _ms, False)),
            _kpi("Cost per run", _usd(s["cost_per_run_usd"]), f"{s['steps_per_run']:.1f} steps per run", dl("cost_per_run_usd", _usd, False)),
        ]
    )

    banner = ""
    if d is not None:
        reg = "".join(f'<span class="chip bad">{_e(r["id"])} {_pct(r["before"])} to {_pct(r["after"])}</span>' for r in d.regressed)
        imp = "".join(f'<span class="chip ok">{_e(r["id"])} {_pct(r["before"])} to {_pct(r["after"])}</span>' for r in d.improved)
        head = f'<b>{len(d.regressed)} regressed</b>, <b>{len(d.improved)} improved</b>, {len(d.unchanged)} unchanged versus <b>{_e(baseline.label)}</b>.'
        banner = f'<div class="banner">{head}<div>{reg}{imp}</div></div>'

    base_map = {c.id: c for c in baseline.cases} if baseline else {}
    cases = "".join(_case_block(c, base_map.get(c.id)) for c in result.cases)

    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>evalloop report: {_e(result.suite)} ({_e(result.label)})</title><style>{CSS}</style></head><body><div class=\"wrap\">"
        f'<div class="tag">evalloop report</div><h1>{_e(result.suite)}</h1>'
        f'<div class="sub">run <b>{_e(result.label)}</b> &middot; {_e(result.created)} &middot; {len(result.cases)} cases x {k} trials</div>'
        f'<div class="kpis">{kpis}</div>{banner}<h2>Cases</h2>{cases}'
        '<footer>Generated by evalloop. pass^k means a case counts as reliable only if every trial passes.</footer>'
        "</div></body></html>"
    )
