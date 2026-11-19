"""Explain flaky and failing cases by comparing the trials that passed with the trials that failed.

A case that passes 7 of 10 times usually fails for one concrete reason: a tool that gets skipped,
a wrong argument, an extra call, a step that errors, or a loop. Reading ten traces by hand to find
it is slow. `explain_case` does the comparison for you:

    from evalloop import run_suite, explain_suite
    result = run_suite(agent, cases, trials=10)
    for ex in explain_suite(result):
        print(ex.text())

Every finding is counted, so you can see how strong it is ("in 3 of 3 failing runs, 0 of 7
passing runs"). Findings describe what differs between passing and failing runs; they point at
the likely cause, they do not prove it.
"""
from __future__ import annotations

import dataclasses as dc
import difflib
import json
import statistics
from collections import Counter
from typing import Any, Optional

from .runner import CaseResult, SuiteResult, TrialResult

__all__ = ["Finding", "Divergence", "Explanation", "explain_case", "explain_suite"]


@dc.dataclass
class Finding:
    """One difference between failing and passing runs, with how often it happened."""

    kind: str  # failed_check | skipped_tool | extra_tool | different_args | order | tool_error | more_steps | slower | pattern | answer_only
    summary: str
    fail_hits: int
    fail_total: int
    pass_hits: int = 0
    pass_total: int = 0
    example: str = ""

    @property
    def strength(self) -> float:
        """How much more common this is in failing runs than in passing runs (0 to 1)."""
        f = self.fail_hits / self.fail_total if self.fail_total else 0.0
        p = self.pass_hits / self.pass_total if self.pass_total else 0.0
        return max(0.0, f - p)

    @property
    def counts(self) -> str:
        s = f"{self.fail_hits} of {self.fail_total} failing runs"
        if self.pass_total:
            s += f", {self.pass_hits} of {self.pass_total} passing runs"
        return s


@dc.dataclass
class Divergence:
    """Where one failing run first went a different way from the reference passing run."""

    trial: int
    step: int  # index among tool calls, 0 based
    op: str  # skipped | extra | changed | ended_early
    expected: Optional[str]
    got: Optional[str]

    def text(self) -> str:
        n = self.step + 1
        if self.op == "skipped":
            return f"tool call {n}: expected {self.expected}, but the run skipped it"
        if self.op == "extra":
            return f"tool call {n}: the run made an extra call {self.got}"
        if self.op == "ended_early":
            return f"tool call {n}: expected {self.expected}, but the run stopped calling tools"
        return f"tool call {n}: expected {self.expected}, got {self.got}"


@dc.dataclass
class Row:
    """One row of a side-by-side alignment of a passing run (left) and a failing run (right)."""

    op: str  # same | changed | only_pass | only_fail
    left: Optional[str]
    right: Optional[str]


@dc.dataclass
class Explanation:
    case_id: str
    passes: int
    total: int
    findings: list
    divergences: list
    reference_trial: Optional[int]
    compared_trial: Optional[int]
    alignment: list

    @property
    def fails(self) -> int:
        return self.total - self.passes

    @property
    def headline(self) -> str:
        """The single most useful sentence about why this case fails."""
        checks = [f for f in self.findings if f.kind == "failed_check"]
        causes = [f for f in self.findings if f.kind not in ("failed_check", "pattern", "answer_only")]
        if causes and causes[0].strength >= 0.5:
            return causes[0].summary
        if checks:
            note = " (tool calls matched the passing runs)" if any(f.kind == "answer_only" for f in self.findings) else ""
            return checks[0].summary + note
        if self.findings:
            return self.findings[0].summary
        return "no difference found in tool calls; compare the answers"

    def text(self) -> str:
        state = "flaky" if self.passes else "always failing"
        lines = [f"{self.case_id}  {self.passes}/{self.total} passed ({state})"]
        for f in self.findings:
            label = {"failed_check": "symptom", "pattern": "what it did", "answer_only": "where to look"}.get(f.kind) or ("likely cause" if f.strength >= 0.5 else "also differs")
            lines.append(f"  {label:<13} {f.summary}  [{f.counts}]")
        if self.divergences:
            d = self.divergences[0]
            ref = f" vs passing trial {self.reference_trial + 1}" if self.reference_trial is not None else ""
            lines.append(f"  first split   trial {d.trial + 1}{ref}: {d.text()}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = dc.asdict(self)
        d["headline"] = self.headline
        return d


# ---------------------------------------------------------------- helpers

def _canon(x: Any, n: int = 80) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        s = x
    else:
        try:
            s = json.dumps(x, sort_keys=True, default=str, separators=(",", ":"))
        except (TypeError, ValueError):
            s = str(x)
    return s if len(s) <= n else s[: n - 3] + "..."


def _sig(step) -> str:
    return f"{step.name}({_canon(step.input, 60)})"


def _compress(names: list) -> str:
    out, i = [], 0
    while i < len(names):
        j = i
        while j < len(names) and names[j] == names[i]:
            j += 1
        out.append(names[i] if j - i == 1 else f"{names[i]} x{j - i}")
        i = j
    return ", ".join(out)


def _tools(t: TrialResult) -> list:
    return [s for s in t.trace.steps if s.kind == "tool"]


def _args(step) -> dict:
    return step.input if isinstance(step.input, dict) else {}


def _rate(trials: list, pred) -> int:
    return sum(1 for t in trials if pred(t))


def _reference(passing: list) -> Optional[TrialResult]:
    """The passing run whose tool sequence is the most common one among passing runs."""
    if not passing:
        return None
    seqs = Counter(tuple(s.name for s in _tools(t)) for t in passing)
    best = seqs.most_common(1)[0][0]
    return next(t for t in passing if tuple(s.name for s in _tools(t)) == best)


def _first_divergence(ref: TrialResult, run: TrialResult) -> Optional[Divergence]:
    a = [_sig(s) for s in _tools(ref)]
    b = [_sig(s) for s in _tools(run)]
    an = [s.name for s in _tools(ref)]
    bn = [s.name for s in _tools(run)]
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            op = "ended_early" if j1 >= len(b) else "skipped"
            return Divergence(run.trial, i1, op, a[i1], None)
        if tag == "insert":
            return Divergence(run.trial, j1, "extra", None, b[j1])
        # replace: same tool with other args, or a different tool
        op = "changed"
        if an[i1:i2] == bn[j1:j2]:
            op = "changed"
        return Divergence(run.trial, i1, op, a[i1], b[j1])
    return None


def _align(ref: TrialResult, run: TrialResult) -> list:
    a, b = _tools(ref), _tools(run)
    sa, sb = [_sig(s) for s in a], [_sig(s) for s in b]
    rows = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=sa, b=sb, autojunk=False).get_opcodes():
        if tag == "equal":
            rows += [Row("same", sa[i], sb[j]) for i, j in zip(range(i1, i2), range(j1, j2))]
        elif tag == "delete":
            rows += [Row("only_pass", sa[i], None) for i in range(i1, i2)]
        elif tag == "insert":
            rows += [Row("only_fail", None, sb[j]) for j in range(j1, j2)]
        else:
            la, lb = list(range(i1, i2)), list(range(j1, j2))
            for k in range(max(len(la), len(lb))):
                l = sa[la[k]] if k < len(la) else None
                r = sb[lb[k]] if k < len(lb) else None
                rows.append(Row("changed" if l and r else ("only_pass" if l else "only_fail"), l, r))
    rows.append(Row("same" if ref.trace.answer == run.trace.answer else "changed",
                    "answer: " + _canon(ref.trace.answer, 120), "answer: " + _canon(run.trace.answer, 120)))
    return rows


# ---------------------------------------------------------------- main entry points

def explain_case(case: CaseResult) -> Optional[Explanation]:
    """Compare the passing and failing trials of one case. Returns None if every trial passed."""
    passing = [t for t in case.trials if t.passed]
    failing = [t for t in case.trials if not t.passed]
    if not failing:
        return None
    P, F = len(passing), len(failing)
    findings: list = []

    # 1. what the checks said (the symptom)
    failed = Counter()
    detail: dict = {}
    for t in failing:
        for c in t.checks:
            if not c.passed:
                failed[c.name] += 1
                detail.setdefault(c.name, c.detail)
    for name, n in failed.most_common():
        ex = f": {detail[name]}" if detail.get(name) else ""
        findings.append(Finding("failed_check", f"check {name} failed{ex}", n, F, example=detail.get(name, "")))

    causes: list = []
    names = sorted({s.name for t in case.trials for s in _tools(t)})

    # 2. a tool that passing runs call and failing runs skip, or the other way round
    for name in names:
        fp = _rate(passing, lambda t: any(s.name == name for s in _tools(t)))
        ff = _rate(failing, lambda t: any(s.name == name for s in _tools(t)))
        if P and fp / P - ff / F >= 0.5:
            causes.append(Finding("skipped_tool", f"skipped {name}", F - ff, F, P - fp, P))
        elif ff / F - (fp / P if P else 0) >= 0.5:
            causes.append(Finding("extra_tool", f"called {name}, which passing runs do not call" if P else f"called {name}", ff, F, fp, P))

    # 3. same tool, different argument values
    for name in names:
        pass_vals: dict = {}
        for t in passing:
            for s in _tools(t):
                if s.name == name:
                    for k, v in _args(s).items():
                        pass_vals.setdefault(k, Counter())[_canon(v, 40)] += 1
        if not pass_vals:
            continue
        for key, cnt in pass_vals.items():
            usual = cnt.most_common(1)[0][0]
            if len(cnt) > 1 and cnt.most_common(1)[0][1] < 0.8 * sum(cnt.values()):
                continue  # passing runs do not agree on this argument, so it is not a signal
            hits, seen = 0, None
            for t in failing:
                vals = [_canon(_args(s).get(key), 40) for s in _tools(t) if s.name == name and key in _args(s)]
                if vals and any(v != usual for v in vals):
                    hits += 1
                    seen = seen or next(v for v in vals if v != usual)
            if hits and hits / F >= 0.5:
                causes.append(Finding("different_args", f"{name} got {key}={seen} (passing runs used {usual})", hits, F, 0, P))

    # 4. same tools, different order
    if P:
        ref_order = Counter(tuple(s.name for s in _tools(t)) for t in passing).most_common(1)[0][0]
        hits = 0
        for t in failing:
            seq = tuple(s.name for s in _tools(t))
            if seq != ref_order and sorted(seq) == sorted(ref_order):
                hits += 1
        if hits:
            causes.append(Finding("order", f"called the same tools in a different order (passing order: {', '.join(ref_order)})", hits, F, 0, P))

    # 5. tool errors
    ef = _rate(failing, lambda t: any(s.error for s in _tools(t)))
    ep = _rate(passing, lambda t: any(s.error for s in _tools(t)))
    if ef:
        err = next(s for t in failing for s in _tools(t) if s.error)
        causes.append(Finding("tool_error", f"a tool call errored ({err.name}: {_canon(err.error, 60)})", ef, F, ep, P))

    # 6. more steps than usual (often a loop) and slower runs
    if P:
        mp = statistics.median(len(_tools(t)) for t in passing)
        mf = statistics.median(len(_tools(t)) for t in failing)
        if mf >= mp + 2:
            hits = _rate(failing, lambda t: len(_tools(t)) >= mp + 2)
            causes.append(Finding("more_steps", f"made more tool calls (median {mf:g} vs {mp:g}), check for a loop", hits, F, 0, P))
        lp = statistics.median(t.trace.latency_ms for t in passing)
        lf = statistics.median(t.trace.latency_ms for t in failing)
        if lp > 0 and lf > 1.5 * lp and lf - lp > 200:
            causes.append(Finding("slower", f"slower (median {lf / 1000:.1f}s vs {lp / 1000:.1f}s)", _rate(failing, lambda t: t.trace.latency_ms > 1.5 * lp), F, 0, P))

    if not P:
        # nothing to compare against: describe what every failing run did instead
        causes = [c for c in causes if c.kind not in ("extra_tool",)]
        seqs = Counter(_compress([s.name for s in _tools(t)]) for t in failing)
        seq, n = seqs.most_common(1)[0]
        causes.append(Finding("pattern", f"no run passed; the usual tool sequence was: {seq or 'no tool calls'}", n, F))
    causes.sort(key=lambda f: (-f.strength, -f.fail_hits))
    if P and not any(c.strength >= 0.5 for c in causes):
        same = _rate(failing, lambda t: tuple(_sig(s) for s in _tools(t)) in {tuple(_sig(s) for s in _tools(x)) for x in passing})
        if same:
            causes.insert(0, Finding("answer_only", "made the same tool calls as a passing run; the difference is in the final answer", same, F, P, P))
    findings += causes

    ref = _reference(passing)
    divs = []
    if ref is not None:
        for t in failing:
            d = _first_divergence(ref, t)
            if d:
                divs.append(d)
    compared = failing[0]
    if divs:
        compared = next(t for t in failing if t.trial == divs[0].trial)
    return Explanation(
        case_id=case.id,
        passes=P,
        total=P + F,
        findings=findings,
        divergences=divs,
        reference_trial=ref.trial if ref else None,
        compared_trial=compared.trial,
        alignment=_align(ref, compared) if ref else [],
    )


def explain_suite(result: SuiteResult, only_flaky: bool = False) -> list:
    """Explanations for every case that failed at least once (flaky first)."""
    out = []
    for c in result.cases:
        if only_flaky and not c.flaky:
            continue
        ex = explain_case(c)
        if ex:
            out.append(ex)
    out.sort(key=lambda e: (e.passes == 0, -e.fails))
    return out
