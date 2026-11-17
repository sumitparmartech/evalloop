"""Bring in traces you already have, and test them without calling a model again.

Use this to run evalloop checks on real production conversations (for example
logged OpenAI or Anthropic message lists), to build a regression set from failures
you saw in the wild, or to save traces as JSONL.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

from .core import Step, Trace


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def from_openai_messages(messages: list) -> Trace:
    """Build a Trace from an OpenAI-style message list (also Groq, OpenRouter, vLLM, Ollama)."""
    user = next((m.get("content") or "" for m in messages if m.get("role") == "user"), "")
    trace = Trace(input=user if isinstance(user, str) else json.dumps(user))
    names: dict = {}
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            for c in m.get("tool_calls") or []:
                fn = c.get("function", {})
                names[c.get("id")] = len(trace.steps)
                trace.steps.append(Step("tool", fn.get("name", "?"), _loads(fn.get("arguments") or "{}")))
            if m.get("content"):
                trace.answer = m["content"]
        elif role == "tool":
            idx = names.get(m.get("tool_call_id"))
            out = _loads(m.get("content"))
            if idx is not None:
                trace.steps[idx].output = out
                if isinstance(out, dict) and out.get("error"):
                    trace.steps[idx].error = str(out["error"])
    return trace


def from_anthropic_messages(messages: list) -> Trace:
    """Build a Trace from an Anthropic-style message list (tool_use / tool_result blocks)."""
    first = next((m.get("content") for m in messages if m.get("role") == "user"), "")
    if isinstance(first, list):
        first = "".join(b.get("text", "") for b in first if b.get("type") == "text")
    trace = Trace(input=first or "")
    index: dict = {}
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            if m.get("role") == "assistant":
                trace.answer = content
            continue
        for b in content or []:
            kind = b.get("type")
            if kind == "tool_use":
                index[b.get("id")] = len(trace.steps)
                trace.steps.append(Step("tool", b.get("name", "?"), b.get("input")))
            elif kind == "tool_result":
                i = index.get(b.get("tool_use_id"))
                out = b.get("content")
                if isinstance(out, list):
                    out = "".join(x.get("text", "") for x in out if x.get("type") == "text")
                out = _loads(out)
                if i is not None:
                    trace.steps[i].output = out
                    if b.get("is_error") or (isinstance(out, dict) and out.get("error")):
                        trace.steps[i].error = str(out.get("error") if isinstance(out, dict) else out)
            elif kind == "text" and m.get("role") == "assistant":
                trace.answer = b.get("text", "")
    return trace


def save_traces(path: str, traces: Iterable[Trace]) -> int:
    """Write traces as JSON Lines. Returns how many were written."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), default=str) + "\n")
            n += 1
    return n


def load_traces(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [Trace.from_dict(json.loads(line)) for line in f if line.strip()]


def replay_agent(traces: Iterable[Trace]):
    """An agent that returns recorded traces instead of calling a model.

    Match is by exact user input. Use it with run_suite to put production traces
    through your checks:  run_suite(replay_agent(traces), cases)
    """
    by_input = {t.input: t for t in traces}

    def agent(text: str, ctx: Any = None) -> Trace:
        if text not in by_input:
            raise KeyError(f"no recorded trace for input {text!r}")
        return by_input[text]

    return agent
