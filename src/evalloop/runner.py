"""Suites, the runner, result models, metrics and run-to-run diffing."""
from __future__ import annotations

import dataclasses as dc
import datetime as dt
import hashlib
import json
from typing import Any, Callable, Optional

from .checks import Check, CheckResult
from .core import Trace


@dc.dataclass
class Case:
    id: str
    input: str
    checks: list = dc.field(default_factory=list)
    tags: list = dc.field(default_factory=list)


@dc.dataclass
class Context:
    """Passed to your agent so runs are reproducible."""

    seed: int
    trial: int
    case: Case


@dc.dataclass
class TrialResult:
    trial: int
    passed: bool
    checks: list
    trace: Trace


@dc.dataclass
class CaseResult:
    id: str
    input: str
    tags: list
    trials: list

    @property
    def pass_rate(self) -> float:
        return sum(t.passed for t in self.trials) / len(self.trials) if self.trials else 0.0

    @property
    def pass_all(self) -> bool:
        """pass^k: every one of the k trials passed."""
        return bool(self.trials) and all(t.passed for t in self.trials)

    @property
    def pass_any(self) -> bool:
        """pass@k: at least one of the k trials passed."""
        return any(t.passed for t in self.trials)


@dc.dataclass
class SuiteResult:
    suite: str
    label: str
    trials_per_case: int
    created: str
    cases: list
    meta: dict = dc.field(default_factory=dict)

    # ---- metrics
    def _all_traces(self) -> list:
        return [t.trace for c in self.cases for t in c.trials]

    @property
    def pass_rate(self) -> float:
        runs = [t.passed for c in self.cases for t in c.trials]
        return sum(runs) / len(runs) if runs else 0.0

    @property
    def reliability(self) -> float:
        """Share of cases that passed on every trial (pass^k)."""
        return sum(c.pass_all for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def latency_p50(self) -> float:
        return _pct([t.latency_ms for t in self._all_traces()], 50)

    @property
    def latency_p95(self) -> float:
        return _pct([t.latency_ms for t in self._all_traces()], 95)

    @property
    def cost_per_run(self) -> float:
        tr = self._all_traces()
        return sum(t.cost for t in tr) / len(tr) if tr else 0.0

    @property
    def steps_per_run(self) -> float:
        tr = self._all_traces()
        return sum(len(t.steps) for t in tr) / len(tr) if tr else 0.0

    def summary(self) -> dict:
        return {
            "pass_rate": self.pass_rate,
            "reliability": self.reliability,
            "latency_p50_ms": self.latency_p50,
            "latency_p95_ms": self.latency_p95,
            "cost_per_run_usd": self.cost_per_run,
            "steps_per_run": self.steps_per_run,
        }

    # ---- io
    def to_dict(self) -> dict:
        d = dc.asdict(self)
        d["summary"] = self.summary()
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "SuiteResult":
        cases = []
        for c in d["cases"]:
            trials = [
                TrialResult(
                    trial=t["trial"],
                    passed=t["passed"],
                    checks=[CheckResult(**ck) for ck in t["checks"]],
                    trace=Trace.from_dict(t["trace"]),
                )
                for t in c["trials"]
            ]
            cases.append(CaseResult(c["id"], c["input"], c.get("tags", []), trials))
        return cls(d["suite"], d["label"], d["trials_per_case"], d["created"], cases, d.get("meta", {}))

    @classmethod
    def load(cls, path: str) -> "SuiteResult":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _pct(values: list, p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def _seed(case_id: str, trial: int, base: int) -> int:
    h = hashlib.sha256(f"{base}:{case_id}:{trial}".encode()).hexdigest()
    return int(h[:8], 16)


def run_suite(
    agent: Callable[[str, Context], Any],
    cases: list,
    trials: int = 1,
    label: str = "run",
    suite: str = "suite",
    seed: int = 0,
    meta: Optional[dict] = None,
) -> SuiteResult:
    """Run every case `trials` times and evaluate all checks on each trace.

    `agent(input, ctx)` must return a Trace (use Tracer) or a plain string answer.
    """
    results = []
    for case in cases:
        trial_results = []
        for t in range(trials):
            ctx = Context(seed=_seed(case.id, t, seed), trial=t, case=case)
            try:
                out = agent(case.input, ctx)
                trace = out if isinstance(out, Trace) else Trace(case.input, answer=str(out))
            except Exception as exc:
                trace = Trace(case.input, error=f"{type(exc).__name__}: {exc}")
            if trace.error:
                checks = [CheckResult("agent_completed", False, trace.error)]
            else:
                checks = [c(trace, case) for c in case.checks]
            trial_results.append(TrialResult(t, all(c.passed for c in checks), checks, trace))
        results.append(CaseResult(case.id, case.input, list(case.tags), trial_results))
    return SuiteResult(
        suite=suite,
        label=label,
        trials_per_case=trials,
        created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        cases=results,
        meta=meta or {},
    )


@dc.dataclass
class Diff:
    regressed: list
    improved: list
    unchanged: list
    added: list
    removed: list
    deltas: dict

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressed)


def diff_runs(base: SuiteResult, new: SuiteResult) -> Diff:
    """Compare two runs case by case. A regression is any drop in pass rate."""
    b = {c.id: c for c in base.cases}
    n = {c.id: c for c in new.cases}
    regressed, improved, unchanged = [], [], []
    for cid, nc in n.items():
        if cid not in b:
            continue
        before, after = b[cid].pass_rate, nc.pass_rate
        row = {"id": cid, "before": before, "after": after}
        if after < before - 1e-9:
            regressed.append(row)
        elif after > before + 1e-9:
            improved.append(row)
        else:
            unchanged.append(row)
    sb, sn = base.summary(), new.summary()
    deltas = {k: sn[k] - sb[k] for k in sn}
    return Diff(
        regressed=regressed,
        improved=improved,
        unchanged=unchanged,
        added=[c for c in n if c not in b],
        removed=[c for c in b if c not in n],
        deltas=deltas,
    )
