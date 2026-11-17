"""Test agents built with popular frameworks, without changing them.

    from evalloop.frameworks import openai_agents_agent, langgraph_agent

    run_suite(openai_agents_agent(my_agent), cases)     # OpenAI Agents SDK
    run_suite(langgraph_agent(my_graph), cases)         # LangGraph / LangChain

Both wrappers run your agent as it is, then turn what happened (tool calls,
tool outputs, model calls, token usage, final answer) into an evalloop Trace,
so every check, report and CI gate works the same way.

No framework is a hard dependency: the converters read attributes by name, so
they also accept plain dicts (for example traces you logged earlier).

Prompt injection tests: inside a tool, return `poisoned("tool_name", result)`
and evalloop adds the attack payload when a Case sets `poison=`.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

from .core import PRICE_IN, PRICE_OUT, Step, Trace, text_of
from .instrument import apply_poison as poisoned

__all__ = [
    "from_agents_result", "openai_agents_agent",
    "from_langchain_messages", "langgraph_agent", "poisoned",
]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _llm_step(name: str, tin: int, tout: int, ms: float, price_in: float, price_out: float, output: Any = None) -> Step:
    cost = tin * price_in / 1e6 + tout * price_out / 1e6
    return Step("llm", name, None, output, ms, tin, tout, round(cost, 8))


def _history_messages(ctx: Any) -> list:
    out = []
    for said, answered in getattr(ctx, "history", None) or []:
        out += [{"role": "user", "content": said}, {"role": "assistant", "content": answered}]
    return out


# ---------------------------------------------------------------- OpenAI Agents SDK

def _message_text(raw: Any) -> str:
    parts = _get(raw, "content") or []
    if isinstance(parts, str):
        return parts
    return "".join(_get(p, "text", "") or "" for p in parts)


def from_agents_result(result: Any, input: Optional[str] = None, *, latency_ms: float = 0.0,
                       price_in: float = PRICE_IN, price_out: float = PRICE_OUT) -> Trace:
    """Turn an OpenAI Agents SDK RunResult (from Runner.run / run_sync) into a Trace."""
    if input is None:
        raw_in = _get(result, "input", "")
        if isinstance(raw_in, list):
            users = [m for m in raw_in if _get(m, "role") == "user"]
            raw_in = _get(users[-1], "content", "") if users else ""
        input = raw_in if isinstance(raw_in, str) else text_of(raw_in)
    trace = Trace(input=input)

    responses = list(_get(result, "raw_responses", None) or [])
    per_call = latency_ms / len(responses) if responses else 0.0
    for i, r in enumerate(responses):
        usage = _get(r, "usage")
        tin = int(_get(usage, "input_tokens", 0) or 0)
        tout = int(_get(usage, "output_tokens", 0) or 0)
        trace.steps.append(_llm_step(f"model_call_{i + 1}", tin, tout, per_call, price_in, price_out))

    by_call: dict = {}
    answer = ""
    for item in _get(result, "new_items", None) or []:
        kind = _get(item, "type", "")
        raw = _get(item, "raw_item")
        if kind == "tool_call_item":
            name = _get(raw, "name") or _get(item, "_resolved_tool_name") or _get(raw, "type", "tool")
            args = _loads(_get(raw, "arguments", "{}"))
            by_call[_get(raw, "call_id") or _get(raw, "id")] = len(trace.steps)
            trace.steps.append(Step("tool", name, args))
        elif kind == "tool_call_output_item":
            out = _loads(_get(item, "output", _get(raw, "output")))
            idx = by_call.get(_get(raw, "call_id"))
            if idx is not None:
                trace.steps[idx].output = out
                if isinstance(out, dict) and out.get("error"):
                    trace.steps[idx].error = str(out["error"])
            else:
                trace.steps.append(Step("tool", "unknown_tool", None, out))
        elif kind == "message_output_item":
            answer = _message_text(raw) or answer

    final = _get(result, "final_output")
    if final is not None:
        answer = final if isinstance(final, str) else text_of(final)
    trace.answer = answer or ""
    return trace


def openai_agents_agent(agent: Any, *, max_turns: int = 10, run_config: Any = None,
                        price_in: float = PRICE_IN, price_out: float = PRICE_OUT) -> Callable:
    """Wrap an OpenAI Agents SDK `Agent` so run_suite can call it.

    Multi-turn cases pass the earlier turns to the agent as message history.
    """
    try:
        from agents import Runner  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("openai_agents_agent needs the OpenAI Agents SDK: pip install openai-agents") from exc

    def run(text: str, ctx: Any = None) -> Trace:
        history = _history_messages(ctx)
        payload: Any = history + [{"role": "user", "content": text}] if history else text
        kwargs: dict = {"max_turns": max_turns}
        if run_config is not None:
            kwargs["run_config"] = run_config
        start = time.perf_counter()
        result = Runner.run_sync(agent, payload, **kwargs)
        ms = (time.perf_counter() - start) * 1000
        return from_agents_result(result, text, latency_ms=ms, price_in=price_in, price_out=price_out)

    return run


# ---------------------------------------------------------------- LangGraph / LangChain

def _role(m: Any) -> str:
    if isinstance(m, (tuple, list)) and len(m) == 2:
        return {"human": "user", "ai": "assistant"}.get(m[0], m[0])
    r = _get(m, "role") or _get(m, "type") or ""
    return {"human": "user", "ai": "assistant", "AIMessageChunk": "assistant"}.get(r, r)


def _content(m: Any) -> str:
    if isinstance(m, (tuple, list)) and len(m) == 2:
        return str(m[1])
    c = _get(m, "content", "")
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return c or ""


def from_langchain_messages(messages: list, input: Optional[str] = None, *, latency_ms: float = 0.0,
                            price_in: float = PRICE_IN, price_out: float = PRICE_OUT) -> Trace:
    """Turn a LangChain / LangGraph message list (or the same as dicts) into a Trace."""
    if input is None:
        users = [m for m in messages if _role(m) == "user"]
        input = _content(users[0]) if users else ""
    trace = Trace(input=input)
    ai = [m for m in messages if _role(m) == "assistant"]
    per_call = latency_ms / len(ai) if ai else 0.0
    by_id: dict = {}
    n = 0
    for m in messages:
        role = _role(m)
        if role == "assistant":
            n += 1
            usage = _get(m, "usage_metadata") or {}
            calls = _get(m, "tool_calls") or []
            text = _content(m)
            tin = int(_get(usage, "input_tokens", 0) or 0)
            tout = int(_get(usage, "output_tokens", 0) or 0)
            trace.steps.append(_llm_step(f"model_call_{n}", tin, tout, per_call, price_in, price_out,
                                         output=text or [(_get(c, "name") or _get(_get(c, "function"), "name")) for c in calls]))
            for c in calls:
                fn = _get(c, "function")
                name = _get(c, "name") or _get(fn, "name", "?")
                args = _get(c, "args")
                if args is None:
                    args = _loads(_get(fn, "arguments", "{}"))
                by_id[_get(c, "id")] = len(trace.steps)
                trace.steps.append(Step("tool", name, args))
            if text and not calls:
                trace.answer = text
        elif role == "tool":
            out = _loads(_content(m))
            idx = by_id.get(_get(m, "tool_call_id"))
            if idx is None:
                trace.steps.append(Step("tool", _get(m, "name") or "unknown_tool", None, out))
                idx = len(trace.steps) - 1
            else:
                trace.steps[idx].output = out
            if _get(m, "status") == "error" or (isinstance(out, dict) and out.get("error")):
                trace.steps[idx].error = str(out.get("error") if isinstance(out, dict) else out)
    return trace


def langgraph_agent(graph: Any, *, input_key: str = "messages", config: Optional[dict] = None,
                    price_in: float = PRICE_IN, price_out: float = PRICE_OUT) -> Callable:
    """Wrap a compiled LangGraph graph (or any runnable with .invoke) whose state holds messages.

    Only the messages produced during this call count toward the trace.
    """

    def run(text: str, ctx: Any = None) -> Trace:
        sent = _history_messages(ctx) + [{"role": "user", "content": text}]
        start = time.perf_counter()
        state = graph.invoke({input_key: sent}, config=config) if config else graph.invoke({input_key: sent})
        ms = (time.perf_counter() - start) * 1000
        msgs = list(_get(state, input_key) or [])
        new = msgs[len(sent):] if len(msgs) >= len(sent) else msgs
        return from_langchain_messages(new, text, latency_ms=ms, price_in=price_in, price_out=price_out)

    return run
