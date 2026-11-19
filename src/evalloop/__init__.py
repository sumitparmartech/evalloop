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
    judge,
    latency_under,
    max_steps,
    no_loop,
    no_pii,
    no_secret_leak,
    no_tool_errors,
    tool_args,
    tool_called,
    tool_count,
    tool_not_called,
    tool_order,
)
from .adapters import AdapterError, anthropic_agent, chat_agent, from_provider, judge_with
from .core import Step, Trace, Tracer
from .report import render_html
from .traces import from_anthropic_messages, from_openai_messages, load_traces, replay_agent, save_traces
from .ci import assert_reliable, to_junit, to_markdown
from .redteam import INJECTIONS, injection_cases, perturb, variants
from .instrument import agent, current, log_llm, poisoning, record, tool
from .instrument import apply_poison as poisoned
from .runner import Case, Context, Conversation, Turn, SuiteResult, diff_runs, run_suite
from .explain import explain_case, explain_suite
from .compare import compare_runs, render_compare_html

__version__ = "0.7.0"

__all__ = [
    "AdapterError", "poisoned", "explain_case", "explain_suite", "compare_runs", "render_compare_html", "INJECTIONS", "injection_cases", "perturb", "variants", "agent", "tool", "log_llm", "record", "current", "poisoning", "Case", "Conversation", "Turn", "Check", "CheckResult", "Context", "Step", "SuiteResult", "Trace", "Tracer",
    "answer_contains", "answer_matches", "answer_not_contains", "cost_under", "custom",
    "chat_agent", "diff_runs", "from_provider", "grounded", "json_answer", "judge", "latency_under", "max_steps", "no_loop",
    "no_pii", "no_secret_leak", "no_tool_errors", "render_html", "run_suite", "tool_args",
    "tool_called", "tool_count", "tool_not_called", "tool_order",
    "anthropic_agent", "judge_with", "from_openai_messages", "from_anthropic_messages",
    "load_traces", "save_traces", "replay_agent", "to_junit", "to_markdown", "assert_reliable",
]
