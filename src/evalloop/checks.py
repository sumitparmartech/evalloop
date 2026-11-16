"""Built-in checks. Each factory returns a Check you can attach to a Case.

A check function receives (trace, case) and returns either a bool or a
(bool, detail) tuple. Raising inside a check counts as a failure.
"""
from __future__ import annotations

import dataclasses as dc
import json
import re
from typing import Any, Callable

from .core import Trace, text_of


@dc.dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dc.dataclass(frozen=True)
class Check:
    name: str
    fn: Callable[[Trace, Any], Any]

    def __call__(self, trace: Trace, case: Any) -> CheckResult:
        try:
            out = self.fn(trace, case)
        except Exception as exc:
            return CheckResult(self.name, False, f"check raised {type(exc).__name__}: {exc}")
        if isinstance(out, tuple):
            ok, detail = out
            return CheckResult(self.name, bool(ok), str(detail))
        return CheckResult(self.name, bool(out), "")


def custom(name: str, fn: Callable[[Trace, Any], Any]) -> Check:
    """Wrap any function as a check."""
    return Check(name, fn)


# ---------------------------------------------------------------- tool use


def tool_called(name: str, times: int | None = None) -> Check:
    def fn(trace, case):
        n = sum(1 for s in trace.tool_calls if s.name == name)
        ok = n > 0 if times is None else n == times
        return ok, f"{name} called {n}x" + (f", expected {times}x" if times is not None else "")

    return Check(f"tool_called({name})" + (f" x{times}" if times is not None else ""), fn)


def tool_not_called(name: str) -> Check:
    def fn(trace, case):
        n = sum(1 for s in trace.tool_calls if s.name == name)
        return n == 0, f"{name} called {n}x"

    return Check(f"tool_not_called({name})", fn)


def tool_args(name: str, **expected: Any) -> Check:
    """Some call to `name` must contain all the expected argument values."""

    def fn(trace, case):
        calls = [s for s in trace.tool_calls if s.name == name]
        if not calls:
            return False, f"{name} was never called"
        for s in calls:
            args = s.input if isinstance(s.input, dict) else {}
            if all(k in args and _same(args[k], v) for k, v in expected.items()):
                return True, ""
        seen = [s.input for s in calls]
        return False, f"expected {expected}, saw {seen}"

    return Check(f"tool_args({name})", fn)


def tool_order(*names: str) -> Check:
    """These tools must appear in this relative order (other calls may sit between)."""

    def fn(trace, case):
        seq = [s.name for s in trace.tool_calls]
        i = 0
        for n in seq:
            if i < len(names) and n == names[i]:
                i += 1
        return i == len(names), f"saw {seq}"

    return Check("tool_order(" + " > ".join(names) + ")", fn)


def no_loop(max_repeats: int = 2) -> Check:
    """Fail if the same tool is called with the same args more than `max_repeats` times."""

    def fn(trace, case):
        counts: dict[str, int] = {}
        for s in trace.tool_calls:
            key = s.name + text_of(s.input)
            counts[key] = counts.get(key, 0) + 1
        worst = max(counts.values(), default=0)
        return worst <= max_repeats, f"most repeated identical call: {worst}x"

    return Check(f"no_loop(<= {max_repeats})", fn)


def no_tool_errors() -> Check:
    def fn(trace, case):
        errs = [f"{s.name}: {s.error}" for s in trace.tool_calls if s.error]
        return not errs, "; ".join(errs)

    return Check("no_tool_errors", fn)


def max_steps(n: int) -> Check:
    return Check(
        f"max_steps({n})",
        lambda trace, case: (len(trace.steps) <= n, f"{len(trace.steps)} steps"),
    )


# ---------------------------------------------------------------- answers


def answer_contains(*needles: str, any_of: bool = False) -> Check:
    def fn(trace, case):
        low = trace.answer.lower()
        hits = [n for n in needles if n.lower() in low]
        ok = bool(hits) if any_of else len(hits) == len(needles)
        missing = [n for n in needles if n not in hits]
        return ok, "" if ok else f"missing {missing}"

    return Check("answer_contains(" + ", ".join(needles) + ")", fn)


def answer_not_contains(*needles: str) -> Check:
    def fn(trace, case):
        low = trace.answer.lower()
        found = [n for n in needles if n.lower() in low]
        return not found, f"found {found}" if found else ""

    return Check("answer_not_contains(" + ", ".join(needles) + ")", fn)


def answer_matches(pattern: str) -> Check:
    rx = re.compile(pattern, re.I | re.S)
    return Check(
        f"answer_matches({pattern})",
        lambda trace, case: bool(rx.search(trace.answer)),
    )


def json_answer(*keys: str) -> Check:
    """The answer must be a JSON object that contains the given keys."""

    def fn(trace, case):
        try:
            obj = json.loads(trace.answer)
        except ValueError as exc:
            return False, f"not valid JSON: {exc}"
        if not isinstance(obj, dict):
            return False, "JSON is not an object"
        missing = [k for k in keys if k not in obj]
        return not missing, f"missing keys {missing}" if missing else ""

    return Check("json_answer(" + ", ".join(keys) + ")", fn)


# ---------------------------------------------------------------- safety


_NUM = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set:
    out = set()
    for m in _NUM.findall(text):
        try:
            out.add(round(float(m.replace("$", "").replace(",", "")), 4))
        except ValueError:
            pass
    return out


def grounded() -> Check:
    """Every number in the answer must come from the user input or a tool output.

    A cheap, deterministic hallucination detector for agents that report
    amounts, dates, counts, ids and similar facts.
    """

    def fn(trace, case):
        allowed = _numbers(trace.input)
        for s in trace.tool_calls:
            allowed |= _numbers(text_of(s.output))
        bad = sorted(n for n in _numbers(trace.answer) if n not in allowed)
        return not bad, f"numbers not found in any source: {bad}" if bad else ""

    return Check("grounded(numbers)", fn)


def no_secret_leak(*secrets: str) -> Check:
    """None of the secrets (API keys, canary strings) may appear in the answer."""

    def fn(trace, case):
        low = trace.answer.lower()
        leaked = [s for s in secrets if s.lower() in low]
        return not leaked, "LEAKED a secret to the user" if leaked else ""

    return Check("no_secret_leak", fn)


# ---------------------------------------------------------------- budgets


def latency_under(ms: float) -> Check:
    return Check(
        f"latency_under({int(ms)}ms)",
        lambda trace, case: (trace.latency_ms <= ms, f"{trace.latency_ms:.0f} ms"),
    )


def cost_under(usd: float) -> Check:
    return Check(
        f"cost_under(${usd})",
        lambda trace, case: (trace.cost <= usd, f"${trace.cost:.4f}"),
    )


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) < 1e-9
    return str(a) == str(b)
