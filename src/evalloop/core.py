"""Core data model: steps, traces and the Tracer helper used inside agents."""
from __future__ import annotations

import dataclasses as dc
import json
import time
from typing import Any, Callable, Optional

# Default token prices in USD per 1M tokens. Override per call if you need to.
PRICE_IN = 2.50
PRICE_OUT = 10.00


@dc.dataclass
class Step:
    """One thing the agent did: an LLM call or a tool call."""

    kind: str  # "llm" or "tool"
    name: str
    input: Any = None
    output: Any = None
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    error: Optional[str] = None


@dc.dataclass
class Trace:
    """Everything that happened during one run of an agent."""

    input: str
    steps: list = dc.field(default_factory=list)
    answer: str = ""
    error: Optional[str] = None

    @property
    def tool_calls(self) -> list:
        return [s for s in self.steps if s.kind == "tool"]

    @property
    def llm_calls(self) -> list:
        return [s for s in self.steps if s.kind == "llm"]

    @property
    def latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps)

    @property
    def cost(self) -> float:
        return sum(s.cost for s in self.steps)

    @property
    def tokens(self) -> int:
        return sum(s.tokens_in + s.tokens_out for s in self.steps)

    def to_dict(self) -> dict:
        return dc.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        steps = [Step(**s) for s in d.get("steps", [])]
        return cls(input=d["input"], steps=steps, answer=d.get("answer", ""), error=d.get("error"))


def text_of(value: Any) -> str:
    """Stable string form of any tool input or output, used by checks."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _approx_tokens(value: Any) -> int:
    return max(1, len(text_of(value)) // 4)


class Tracer:
    """Records steps while your agent runs. Works with any framework.

    Inside your agent:

        t = Tracer(user_input)
        t.llm("plan", prompt, completion)
        order = t.tool("lookup_order", {"id": 1}, fn=lookup)
        return t.finish("Done")
    """

    def __init__(self, input: str):
        self.trace = Trace(input=input)

    def llm(
        self,
        name: str,
        prompt: Any,
        output: Any,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        latency_ms: float = 0.0,
        price_in: float = PRICE_IN,
        price_out: float = PRICE_OUT,
    ) -> Any:
        tin = _approx_tokens(prompt) if tokens_in is None else tokens_in
        tout = _approx_tokens(output) if tokens_out is None else tokens_out
        cost = tin * price_in / 1e6 + tout * price_out / 1e6
        self.trace.steps.append(
            Step("llm", name, prompt, output, latency_ms, tin, tout, round(cost, 8))
        )
        return output

    def tool(
        self,
        name: str,
        args: Any,
        fn: Optional[Callable[..., Any]] = None,
        output: Any = None,
        latency_ms: Optional[float] = None,
    ) -> Any:
        """Record a tool call. Pass `fn` to run and time it, or `output` to log a result."""
        error = None
        start = time.perf_counter()
        if fn is not None:
            try:
                output = fn(**args) if isinstance(args, dict) else fn(args)
            except Exception as exc:  # the agent sees the error, so we record it
                error = f"{type(exc).__name__}: {exc}"
                output = None
        elapsed = (time.perf_counter() - start) * 1000
        self.trace.steps.append(
            Step("tool", name, args, output, elapsed if latency_ms is None else latency_ms, error=error)
        )
        return output

    def finish(self, answer: str) -> Trace:
        self.trace.answer = answer
        return self.trace
