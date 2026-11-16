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

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def _http_post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:200]
        raise AdapterError(f"HTTP {exc.code}: {body}", exc.code) from None
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
    retries: int = 2,
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

    def call(messages: list) -> dict:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        payload = {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto", "temperature": temperature}
        for attempt in range(retries + 1):
            try:
                return send(url, payload, headers, timeout)
            except AdapterError as exc:
                if exc.status in _RETRY_STATUS and attempt < retries:
                    sleep(2 ** attempt * 2)
                    continue
                raise
        raise AdapterError("unreachable")

    def agent(text: str, ctx: Any = None):
        t = Tracer(text)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
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
