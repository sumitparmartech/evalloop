"""evalloop: unit tests and a CI gate for LLM agents."""
from .checks import (
    Check,
    CheckResult,
    answer_contains,
    answer_matches,
    answer_not_contains,
    cost_under,
    custom,
    grounded,
    json_answer,
    latency_under,
    max_steps,
    no_loop,
    no_secret_leak,
    no_tool_errors,
    tool_args,
    tool_called,
    tool_not_called,
    tool_order,
)
from .adapters import AdapterError, chat_agent, from_provider
from .core import Step, Trace, Tracer
from .report import render_html
from .runner import Case, Context, SuiteResult, diff_runs, run_suite

__version__ = "0.2.0"

__all__ = [
    "AdapterError", "Case", "Check", "CheckResult", "Context", "Step", "SuiteResult", "Trace", "Tracer",
    "answer_contains", "answer_matches", "answer_not_contains", "cost_under", "custom",
    "chat_agent", "diff_runs", "from_provider", "grounded", "json_answer", "latency_under", "max_steps", "no_loop",
    "no_secret_leak", "no_tool_errors", "render_html", "run_suite", "tool_args",
    "tool_called", "tool_not_called", "tool_order",
]
