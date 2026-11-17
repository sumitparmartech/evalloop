"""Suites, the runner, result models, metrics and run-to-run diffing."""
from __future__ import annotations

import concurrent.futures as cf
import contextvars
import dataclasses as dc
import datetime as dt
import hashlib
import json
from typing import Any, Callable, Optional

from .checks import Check, CheckResult
from .core import Trace
from .instrument import poisoning
from .stats import pass_at_k, pass_hat_k, two_proportion_p, wilson_interval


@dc.dataclass
class Turn:
    """One user message in a conversation, with checks that run on that turn only.

    `say` is the text to send, or a function `say(history) -> str` that builds it from
    the conversation so far (a list of (user, answer) pairs), so a scripted user can react
    to what the agent said.
    """

    say: Any
    checks: list = dc.field(default_factory=list)


@dc.dataclass
class Case:
    id: str
    input: str
    checks: list = dc.field(default_factory=list)
    tags: list = dc.field(default_factory=list)
    turns: list = dc.field(default_factory=list)
    # {tool name: attack text}. The text is appended to that tool's result during the run, so you
    # can test prompt injection through tools. Works with @tool and Tracer.tool.
    poison: dict = dc.field(default_factory=dict)


def Conversation(id: str, turns: list, checks: Optional[list] = None, tags: Optional[list] = None,
                 poison: Optional[dict] = None) -> Case:
    """A multi-turn case. `turns` holds Turn objects or plain strings.

    `checks` run once on the whole conversation (all steps, all answers), so tool budgets,
    loops, cost, secret leaks and PII are judged across every turn. Each Turn can also
    carry its own checks.
    """
    ts = [t if isinstance(t, Turn) else Turn(t) for t in turns]
    first = ts[0].say if ts and isinstance(ts[0].say, str) else "(conversation)"
    return Case(id=id, input=first, checks=list(checks or []), tags=list(tags or []), turns=ts, poison=dict(poison or {}))


@dc.dataclass
class Context:
    """Passed to your agent so runs are reproducible."""

    seed: int
    trial: int
    case: Case
    history: list = dc.field(default_factory=list)  # earlier (user, answer) pairs in this conversation
    turn: int = 0  # index of the current turn (0 for single-turn cases)


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

    @property
    def passes(self) -> int:
        return sum(t.passed for t in self.trials)

    @property
    def flaky(self) -> bool:
        """Passed on some trials and failed on others: the most dangerous kind of case."""
        return 0 < self.passes < len(self.trials)

    @property
    def interval(self) -> tuple:
        """95% Wilson confidence interval for this case's true pass rate."""
        return wilson_interval(self.passes, len(self.trials))

    def pass_at(self, k: int) -> float:
        """Unbiased pass@k estimate from the trials that were run."""
        return pass_at_k(len(self.trials), self.passes, k)

    def pass_hat(self, k: int) -> float:
        """Unbiased pass^k estimate (all k trials pass) from the trials that were run."""
        return pass_hat_k(len(self.trials), self.passes, k)


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

    @property
    def flaky_cases(self) -> list:
        return [c.id for c in self.cases if c.flaky]

    @property
    def pass_interval(self) -> tuple:
        """95% confidence interval for the overall pass rate."""
        runs = [t.passed for c in self.cases for t in c.trials]
        return wilson_interval(sum(runs), len(runs))

    def summary(self) -> dict:
        lo, hi = self.pass_interval
        return {
            "pass_rate": self.pass_rate,
            "pass_rate_low": lo,
            "pass_rate_high": hi,
            "flaky_cases": len(self.flaky_cases),
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


def _call_agent(agent, text: str, ctx: Context, timeout: Optional[float]) -> Trace:
    try:
        if timeout:
            # A thread cannot be killed, but a hung agent must not hang the whole suite.
            with cf.ThreadPoolExecutor(max_workers=1) as one:
                fut = one.submit(contextvars.copy_context().run, agent, text, ctx)
                try:
                    out = fut.result(timeout=timeout)
                except cf.TimeoutError:
                    raise TimeoutError(f"agent did not finish within {timeout:g}s") from None
        else:
            out = agent(text, ctx)
        return out if isinstance(out, Trace) else Trace(text, answer=str(out))
    except Exception as exc:
        return Trace(text, error=f"{type(exc).__name__}: {exc}")


def _run_conversation(agent, case: Case, t: int, seed: int, timeout: Optional[float]) -> TrialResult:
    history: list = []
    merged = Trace(input="")
    checks: list = []
    prior: list = []
    for i, turn in enumerate(case.turns):
        text = turn.say(list(history)) if callable(turn.say) else turn.say
        ctx = Context(seed=_seed(f"{case.id}#{i}", t, seed), trial=t, case=case, history=list(history), turn=i)
        tr = _call_agent(agent, text, ctx, timeout)
        merged.turn_inputs.append(text)
        if i == 0:
            merged.input = text
        if tr.error:
            checks.append(CheckResult(f"turn {i + 1}: agent_completed", False, tr.error))
            merged.error = tr.error
            break
        merged.steps.extend(tr.steps)
        merged.turn_answers.append(tr.answer)
        merged.answer = tr.answer
        view = Trace(
            input=text, steps=tr.steps, answer=tr.answer,
            turn_inputs=list(merged.turn_inputs), turn_answers=[tr.answer], prior_steps=list(prior),
        )
        for chk in turn.checks:
            r = chk(view, case)
            checks.append(CheckResult(f"turn {i + 1}: {r.name}", r.passed, r.detail))
        prior.extend(tr.steps)
        history.append((text, tr.answer))
    if not merged.error:
        checks.extend(chk(merged, case) for chk in case.checks)
    return TrialResult(t, all(c.passed for c in checks), checks, merged)


def _run_trial(agent, case: Case, t: int, seed: int, timeout: Optional[float]) -> TrialResult:
    with poisoning(case.poison):
        return _run_trial_inner(agent, case, t, seed, timeout)


def _run_trial_inner(agent, case: Case, t: int, seed: int, timeout: Optional[float]) -> TrialResult:
    if case.turns:
        return _run_conversation(agent, case, t, seed, timeout)
    ctx = Context(seed=_seed(case.id, t, seed), trial=t, case=case)
    trace = _call_agent(agent, case.input, ctx, timeout)
    if trace.error:
        checks = [CheckResult("agent_completed", False, trace.error)]
    else:
        checks = [c(trace, case) for c in case.checks]
    return TrialResult(t, all(c.passed for c in checks), checks, trace)


def run_suite(
    agent: Callable[[str, Context], Any],
    cases: list,
    trials: int = 1,
    label: str = "run",
    suite: str = "suite",
    seed: int = 0,
    meta: Optional[dict] = None,
    workers: int = 1,
    timeout: Optional[float] = None,
    on_trial: Optional[Callable[[Case, TrialResult], None]] = None,
) -> SuiteResult:
    """Run every case `trials` times and evaluate all checks on each trace.

    `agent(input, ctx)` must return a Trace (use Tracer) or a plain string answer.
    workers > 1 runs trials in parallel threads (your agent must be thread-safe, which
    is normal for API calls). timeout is seconds per trial. Results keep their order,
    so a parallel run gives the same report as a sequential one.
    """
    jobs = [(case, t) for case in cases for t in range(trials)]
    done: dict = {}

    def work(job):
        case, t = job
        res = _run_trial(agent, case, t, seed, timeout)
        if on_trial:
            on_trial(case, res)
        return res

    if workers and workers > 1:
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            for job, res in zip(jobs, pool.map(work, jobs)):
                done[(job[0].id, job[1])] = res
    else:
        for job in jobs:
            done[(job[0].id, job[1])] = work(job)

    results = [
        CaseResult(c.id, c.input, list(c.tags), [done[(c.id, t)] for t in range(trials)]) for c in cases
    ]
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


def diff_runs(base: SuiteResult, new: SuiteResult, alpha: Optional[float] = None) -> Diff:
    """Compare two runs case by case.

    By default any drop in a case's pass rate is a regression (strict, good for
    deterministic agents). Pass alpha (for example 0.2) to call a drop a regression
    only when it is statistically unlikely to be noise. Every row carries `p_value`
    and `noise`, so you can see which changes are real and which are just luck.
    """
    b = {c.id: c for c in base.cases}
    n = {c.id: c for c in new.cases}
    regressed, improved, unchanged = [], [], []
    for cid, nc in n.items():
        if cid not in b:
            continue
        bc = b[cid]
        before, after = bc.pass_rate, nc.pass_rate
        p = two_proportion_p(bc.passes, len(bc.trials), nc.passes, len(nc.trials))
        row = {"id": cid, "before": before, "after": after, "p_value": p, "noise": p >= 0.05}
        significant = alpha is None or p < alpha
        if after < before - 1e-9 and significant:
            regressed.append(row)
        elif after > before + 1e-9 and significant:
            improved.append(row)
        else:
            unchanged.append(row)
    sb, sn = base.summary(), new.summary()
    deltas = {k: sn[k] - sb[k] for k in sn}
    runs_b = [t.passed for c in base.cases for t in c.trials]
    runs_n = [t.passed for c in new.cases for t in c.trials]
    deltas["pass_rate_p_value"] = two_proportion_p(sum(runs_b), len(runs_b), sum(runs_n), len(runs_n))
    return Diff(
        regressed=regressed,
        improved=improved,
        unchanged=unchanged,
        added=[c for c in n if c not in b],
        removed=[c for c in b if c not in n],
        deltas=deltas,
    )
