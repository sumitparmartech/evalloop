"""Run a real model through evalloop without extra dependencies.

`chat_agent` builds an agent function for any OpenAI-compatible chat endpoint
(OpenAI, Groq, OpenRouter, Gemini, Ollama, vLLM and others). It runs the usual
tool-calling loop and records every LLM call and tool call with a Tracer, so all
the checks work unchanged.

The API key is read from an environment variable and is only ever sent in the
Authorization header. It is never stored, printed or included in error messages.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from .core import Tracer

# name -> (base url, env var holding the key, a model that is currently offered)
PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.0-flash"),
    "ollama": ("http://localhost:11434/v1", "", "llama3.1"),
}

_RETRY_STATUS = {429, 500, 502, 503, 504}


class AdapterError(RuntimeError):
    """Raised when the provider returns an error. Never contains the API key."""

    def __init__(self, message: str, status: Optional[int] = None, retry_after: Optional[float] = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def _parse_retry_after(header: Optional[str], body: str) -> Optional[float]:
    """Seconds to wait, from a Retry-After header or text like 'try again in 1m2.5s' or 'in 850ms'."""
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            pass
    m = re.search(r"try again in ((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", body or "")
    if not m:
        return None
    total = 0.0
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1)):
        total += float(num) * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return total


def _backoff(exc: "AdapterError", attempt: int) -> float:
    """How long to wait before retrying: what the provider asked for, else exponential."""
    if exc.retry_after is not None:
        return min(90.0, exc.retry_after + 0.5)
    return min(60.0, 2.0 ** attempt * 2)


class _Pacer:
    """Spaces requests so at most `rpm` start per minute, across threads."""

    def __init__(self, rpm: float, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.gap, self.sleep, self.clock = 60.0 / rpm, sleep, clock
        self.next = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = self.clock()
            start = max(now, self.next)
            self.next = start + self.gap
        if start > now:
            self.sleep(start - now)


USER_AGENT = "evalloop/0.7.0 (+https://github.com/sumitparmartech/evalloop)"


def _http_post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    # Some providers sit behind a CDN that rejects the default "Python-urllib" user agent
    # with HTTP 403 (Cloudflare error 1010), so always send a real one.
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json", **headers}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        full = exc.read().decode(errors="replace")
        wait = _parse_retry_after(exc.headers.get("retry-after") if exc.headers else None, full)
        raise AdapterError(f"HTTP {exc.code}: {full[:200]}", exc.code, wait) from None
    except urllib.error.URLError as exc:
        raise AdapterError(f"could not reach {url}: {exc.reason}") from None


def chat_agent(
    *,
    system: str,
    tools: list,
    impls: dict,
    model: str,
    base_url: str = "https://api.openai.com/v1",
    api_key: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    temperature: float = 0.7,
    max_turns: int = 6,
    price_in: float = 0.0,
    price_out: float = 0.0,
    timeout: float = 60.0,
    retries: int = 5,
    rpm: Optional[float] = None,
    post: Optional[Callable[..., dict]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable:
    """Return `agent(text, ctx)` for run_suite.

    tools: OpenAI-format tool schemas. impls: tool name -> python function taking the
    tool arguments as keyword arguments. price_in and price_out are USD per 1M tokens
    (leave 0 for free tiers, so cost checks stay meaningful only when you set them).
    """
    key = api_key if api_key is not None else (os.environ.get(api_key_env, "") if api_key_env else "")
    send = post or _http_post
    url = base_url.rstrip("/") + "/chat/completions"
    gate = _Pacer(rpm, sleep) if rpm else None  # rpm: cap requests per minute (free tiers)

    def call(messages: list) -> dict:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        payload = {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto", "temperature": temperature}
        for attempt in range(retries + 1):
            if gate:
                gate.wait()
            try:
                return send(url, payload, headers, timeout)
            except AdapterError as exc:
                if exc.status in _RETRY_STATUS and attempt < retries:
                    sleep(_backoff(exc, attempt))
                    continue
                raise
        raise AdapterError("unreachable")

    def agent(text: str, ctx: Any = None):
        t = Tracer(text)
        messages = [{"role": "system", "content": system}]
        for said, answered in getattr(ctx, "history", None) or []:
            messages += [{"role": "user", "content": said}, {"role": "assistant", "content": answered}]
        messages.append({"role": "user", "content": text})
        for _ in range(max_turns):
            start = time.perf_counter()
            resp = call(messages)
            ms = (time.perf_counter() - start) * 1000
            msg = resp["choices"][0]["message"]
            usage = resp.get("usage") or {}
            calls = msg.get("tool_calls") or []
            t.llm(
                "plan" if calls else "answer",
                messages[-1].get("content") or "",
                msg.get("content") or [c["function"]["name"] for c in calls],
                tokens_in=usage.get("prompt_tokens"),
                tokens_out=usage.get("completion_tokens"),
                latency_ms=ms,
                price_in=price_in,
                price_out=price_out,
            )
            if not calls:
                return t.finish(msg.get("content") or "")
            messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except ValueError:
                    args = {}
                fn = impls.get(name)
                if fn is None:
                    out = t.tool(name, args, output={"error": f"unknown tool {name}"})
                else:
                    out = t.tool(name, args, fn=fn)
                    if t.trace.steps[-1].error:
                        out = {"error": t.trace.steps[-1].error}
                messages.append({"role": "tool", "tool_call_id": c.get("id", name), "content": json.dumps(out, default=str)})
        trace = t.finish("")
        trace.error = f"no final answer after {max_turns} turns"
        return trace

    return agent


def from_provider(provider: str, *, system: str, tools: list, impls: dict, model: Optional[str] = None, **kw) -> Callable:
    """Shortcut: from_provider("groq", system=..., tools=..., impls=...). The key comes from GROQ_API_KEY."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose one of {', '.join(PROVIDERS)}")
    base, env, default_model = PROVIDERS[provider]
    return chat_agent(system=system, tools=tools, impls=impls, model=model or default_model, base_url=base, api_key_env=env, **kw)


# ---------------------------------------------------------------- Anthropic (native Messages API)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def anthropic_agent(
    *,
    system: str,
    tools: list,
    impls: dict,
    model: str,
    api_key: Optional[str] = None,
    api_key_env: str = "ANTHROPIC_API_KEY",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    max_turns: int = 6,
    price_in: float = 0.0,
    price_out: float = 0.0,
    timeout: float = 60.0,
    retries: int = 2,
    base_url: str = ANTHROPIC_URL,
    post: Optional[Callable[..., dict]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable:
    """Same as chat_agent, but for Claude through the native Messages API.

    tools: Anthropic-format schemas ({"name", "description", "input_schema"}).
    The key is read from ANTHROPIC_API_KEY and only goes in the x-api-key header.
    """
    key = api_key if api_key is not None else os.environ.get(api_key_env, "")
    send = post or _http_post

    def call(messages: list) -> dict:
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if key:
            headers["x-api-key"] = key
        payload = {
            "model": model,
            "system": system,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        for attempt in range(retries + 1):
            try:
                return send(base_url, payload, headers, timeout)
            except AdapterError as exc:
                if (exc.status in _RETRY_STATUS or exc.status == 529) and attempt < retries:
                    sleep(_backoff(exc, attempt))
                    continue
                raise
        raise AdapterError("unreachable")

    def agent(text: str, ctx: Any = None):
        t = Tracer(text)
        messages: list = []
        for said, answered in getattr(ctx, "history", None) or []:
            messages += [{"role": "user", "content": said}, {"role": "assistant", "content": answered}]
        messages.append({"role": "user", "content": text})
        for _ in range(max_turns):
            start = time.perf_counter()
            resp = call(messages)
            ms = (time.perf_counter() - start) * 1000
            blocks = resp.get("content") or []
            uses = [b for b in blocks if b.get("type") == "tool_use"]
            said = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            usage = resp.get("usage") or {}
            t.llm(
                "plan" if uses else "answer",
                messages[-1]["content"] if isinstance(messages[-1]["content"], str) else "tool results",
                said or [u["name"] for u in uses],
                tokens_in=usage.get("input_tokens"),
                tokens_out=usage.get("output_tokens"),
                latency_ms=ms,
                price_in=price_in,
                price_out=price_out,
            )
            if not uses:
                return t.finish(said)
            messages.append({"role": "assistant", "content": blocks})
            results = []
            for u in uses:
                fn = impls.get(u["name"])
                args = u.get("input") or {}
                if fn is None:
                    out = t.tool(u["name"], args, output={"error": f"unknown tool {u['name']}"})
                else:
                    out = t.tool(u["name"], args, fn=fn)
                    if t.trace.steps[-1].error:
                        out = {"error": t.trace.steps[-1].error}
                results.append({"type": "tool_result", "tool_use_id": u["id"], "content": json.dumps(out, default=str)})
            messages.append({"role": "user", "content": results})
        trace = t.finish("")
        trace.error = f"no final answer after {max_turns} turns"
        return trace

    return agent


def judge_with(provider: str, model: Optional[str] = None, *, api_key: Optional[str] = None,
               post: Optional[Callable[..., dict]] = None, timeout: float = 60.0) -> Callable[[str], str]:
    """Build the `ask(prompt) -> text` function that the `judge` check needs.

    provider is "anthropic" or any name in PROVIDERS (openai, groq, openrouter, gemini, ollama).
    Uses temperature 0 so the same answer gets the same score.
    """
    send = post or _http_post
    if provider == "anthropic":
        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")

        def ask_anthropic(prompt: str) -> str:
            headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
            if key:
                headers["x-api-key"] = key
            payload = {"model": model or "claude-haiku-4-5", "max_tokens": 300, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}
            resp = send(ANTHROPIC_URL, payload, headers, timeout)
            return "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")

        return ask_anthropic
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}")
    base, env, default_model = PROVIDERS[provider]
    key = api_key if api_key is not None else (os.environ.get(env, "") if env else "")

    def ask(prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        payload = {"model": model or default_model, "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]}
        resp = send(base.rstrip("/") + "/chat/completions", payload, headers, timeout)
        return resp["choices"][0]["message"].get("content") or ""

    return ask
