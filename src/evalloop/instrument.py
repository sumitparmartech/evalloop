"""Test an agent you already wrote, with two lines of change.

    from evalloop import agent, tool, log_llm

    @tool()
    def lookup_order(order_id): ...        # every call is recorded with args, output, timing, errors

    @agent
    def support_agent(text):               # returns the final answer as a plain string
        ...
        return answer

    run_suite(support_agent, cases)        # no Tracer needed

Works with sync and async functions. Recording uses context variables, so it is safe
with parallel trials. If your agent starts its own threads, wrap the work with
`contextvars.copy_context().run` so the tools are still recorded.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import inspect
import time
from typing import Any, Callable, Dict, Iterator, Optional

from .core import Step, Tracer

_current: contextvars.ContextVar = contextvars.ContextVar("evalloop_tracer", default=None)
_poison: contextvars.ContextVar = contextvars.ContextVar("evalloop_poison", default=None)


def current() -> Optional[Tracer]:
    """The Tracer of the run in progress, or None outside a run."""
    return _current.get()


@contextlib.contextmanager
def record(input: str) -> Iterator[Tracer]:
    """Record everything inside the block:  with record("hi") as t: ...; t.finish(answer)."""
    t = Tracer(input)
    token = _current.set(t)
    try:
        yield t
    finally:
        _current.reset(token)


@contextlib.contextmanager
def poisoning(mapping: Optional[Dict[str, str]]) -> Iterator[None]:
    """Inside the block, outputs of the named tools get the payload appended (see Case.poison)."""
    token = _poison.set(dict(mapping) if mapping else None)
    try:
        yield
    finally:
        _poison.reset(token)


def apply_poison(name: str, output: Any) -> Any:
    """Append an attack payload to a tool result, the way a poisoned web page or document would."""
    payload = (_poison.get() or {}).get(name)
    if not payload:
        return output
    if isinstance(output, str):
        return output + "\n" + payload
    if isinstance(output, dict):
        out = dict(output)
        strings = [k for k, v in out.items() if isinstance(v, str)]
        if strings:
            key = max(strings, key=lambda k: len(out[k]))
            out[key] = out[key] + "\n" + payload
        else:
            out["notes"] = payload
        return out
    if isinstance(output, list):
        return list(output) + [payload]
    return output


def log_llm(name: str, prompt: Any, output: Any, **usage: Any) -> Any:
    """Record one model call from inside an @agent. usage: tokens_in, tokens_out, latency_ms, price_in, price_out."""
    t = current()
    if t is None:
        return output
    return t.llm(name, prompt, output, **usage)


def _bind_args(fn: Callable, args: tuple, kwargs: dict) -> Any:
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        return {"args": list(args), **kwargs}


def tool(name: Optional[str] = None) -> Callable:
    """Record every call to this function as a tool step. Exceptions are recorded and re-raised."""

    def wrap(fn: Callable) -> Callable:
        label = name or fn.__name__

        def _record(args, kwargs, start, out, err):
            t = current()
            if t is None:
                return out
            out = apply_poison(label, out)
            ms = (time.perf_counter() - start) * 1000
            t.trace.steps.append(Step("tool", label, _bind_args(fn, args, kwargs), out, ms, error=err))
            return out

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    out = await fn(*args, **kwargs)
                except Exception as exc:
                    _record(args, kwargs, start, None, f"{type(exc).__name__}: {exc}")
                    raise
                return _record(args, kwargs, start, out, None)

            return awrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                out = fn(*args, **kwargs)
            except Exception as exc:
                _record(args, kwargs, start, None, f"{type(exc).__name__}: {exc}")
                raise
            return _record(args, kwargs, start, out, None)

        return wrapper

    if callable(name):  # used as a bare @tool
        fn, name = name, None
        return wrap(fn)
    return wrap


def agent(fn: Callable) -> Callable:
    """Turn `def my_agent(text) -> str` (or `(text, ctx)`, sync or async) into an evalloop agent.

    The returned function records every @tool call and log_llm call made during the run and
    returns a Trace, so you can pass it straight to run_suite.
    """
    wants_ctx = len([p for p in inspect.signature(fn).parameters.values()
                     if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) >= 2
    is_async = inspect.iscoroutinefunction(fn)

    @functools.wraps(fn)
    def wrapper(text: str, ctx: Any = None):
        with record(text) as t:
            start = time.perf_counter()
            call = (lambda: fn(text, ctx)) if wants_ctx else (lambda: fn(text))
            try:
                out = asyncio.run(call()) if is_async else call()
            except Exception as exc:
                t.trace.error = f"{type(exc).__name__}: {exc}"
                return t.trace
            if not t.trace.llm_calls and not t.trace.steps:
                # nothing instrumented: still report how long the whole agent took
                t.trace.steps.append(Step("llm", "agent", text, str(out), (time.perf_counter() - start) * 1000))
            if hasattr(out, "steps") and hasattr(out, "answer"):  # the agent already returned a Trace
                return out
            return t.finish("" if out is None else str(out))

    return wrapper
