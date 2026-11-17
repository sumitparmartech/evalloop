# evalloop

[![tests](https://github.com/sumitparmartech/evalloop/actions/workflows/tests.yml/badge.svg)](https://github.com/sumitparmartech/evalloop/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/evalloop-agents)](https://pypi.org/project/evalloop-agents/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://pypi.org/project/evalloop-agents/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**[Live demo](https://sumitparmartech.github.io/evalloop/)** · **[How it compares](docs/COMPARISON.md)** · **[Changelog](CHANGELOG.md)**

Unit tests and a CI gate for LLM agents. Zero dependencies, no API key needed to try it. Python 3.9+.

Most agent demos work once. Production agents fail 1 run in 5, quote amounts they never saw, obey instructions hidden in a web page, and loop on vague questions. `evalloop` runs your agent many times against a suite of cases, checks the full trace (not just the final answer), and fails your build when behavior regresses.

![report](https://raw.githubusercontent.com/sumitparmartech/evalloop/main/docs/report_v1.png)

## What it catches

| Failure | Check |
| --- | --- |
| Flaky agent that passes once and fails later | `pass^k` reliability, every case runs k times |
| Hallucinated numbers (amounts, dates, ids) | `grounded()` flags any number not found in the input or a tool result |
| Prompt injection that leaks a secret | `no_secret_leak(...)` |
| Wrong tool, wrong arguments, wrong order | `tool_called`, `tool_not_called`, `tool_args`, `tool_order` |
| Infinite or wasteful tool loops | `no_loop(n)`, `max_steps(n)` |
| Cost and latency creep | `cost_under(usd)`, `latency_under(ms)` |
| PII or keys in the answer | `no_pii()` (email, phone, card, SSN, API key) |
| Expensive tool called too often | `tool_count(name, at_most=n)` |
| Tone, helpfulness, policy (things code cannot check) | `judge(criteria, ask)` LLM-as-judge |
| Agent forgets what the user said two messages ago | `Conversation(...)` multi-turn cases |
| Prompt injection hidden in a tool result | `Case.poison`, `injection_cases(...)` |
| Breaks on typos, SHOUTING, filler words | `variants(case, ...)` |
| Bad output format | `json_answer(...)`, `answer_matches(regex)` |
| Anything else | `custom(name, fn)` |

## Install

```bash
pip install evalloop-agents            # the import name is still: import evalloop
# or the latest code from GitHub:
pip install git+https://github.com/sumitparmartech/evalloop
```

Optional extras: `pip install "evalloop-agents[agents]"` for the OpenAI Agents SDK adapter,
`pip install "evalloop-agents[langgraph]"` for LangGraph.

## Quickstart

```bash
git clone https://github.com/sumitparmartech/evalloop && cd evalloop
pip install -e .
cd examples
python -m evalloop.cli run suite:cases --agent support_agent:agent_v1 \
  --name support-agent --label v1 --trials 5 --seed 2 --out ../docs/run_v1.json
python -m evalloop.cli run suite:cases --agent support_agent:agent_v2 \
  --name support-agent --label v2 --trials 5 --seed 2 --out ../docs/run_v2.json \
  --baseline ../docs/run_v1.json --html ../docs/report.html
```

The example support agent is simulated so the demo is free and deterministic. Version 1 has four real-world bugs, version 2 fixes them:

| | v1 | v2 |
| --- | --- | --- |
| Pass rate | 33% | 100% |
| Reliability (pass^5) | 0% | 100% |
| Latency p95 | 3.1 s | 1.0 s |
| Cost per run | $0.0105 | $0.0026 |

![diff](https://raw.githubusercontent.com/sumitparmartech/evalloop/main/docs/report.png)

## Test your own agent

Wrap any agent (OpenAI, Anthropic, LangChain, your own loop) by recording steps with `Tracer`:

```python
from evalloop import Tracer, Case, run_suite, tool_called, tool_args, grounded, no_loop

def my_agent(user_input, ctx):
    t = Tracer(user_input)
    messages = [{"role": "user", "content": user_input}]
    while True:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        t.llm("step", messages, msg.content or "", resp.usage.prompt_tokens, resp.usage.completion_tokens)
        if not msg.tool_calls:
            return t.finish(msg.content)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = t.tool(call.function.name, args, fn=TOOL_FUNCS[call.function.name])
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

cases = [
    Case("refund_small", "Refund order #1042",
         [tool_args("issue_refund", order_id="1042"), grounded(), no_loop(2)]),
]
result = run_suite(my_agent, cases, trials=5)
print(result.pass_rate, result.reliability)
```

`ctx.seed` is passed to your agent so runs are reproducible when you use seeded randomness.

## Run it on a real model

`chat_agent` and `from_provider` wrap any OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter, Gemini, Ollama, vLLM). They run the tool-calling loop and record every step, so every check works unchanged. There are no extra dependencies.

```bash
export GROQ_API_KEY=...        # a free key from console.groq.com
cd examples
evalloop run real_suite:cases --agent real_agent:agent --trials 3 \
  --name real-support-agent --out run_real.json --html report_real.html
```

```python
from evalloop import from_provider

agent = from_provider("groq", system=SYSTEM, tools=TOOLS, impls={"lookup_order": lookup_order})
# the key is read from GROQ_API_KEY and is never stored, printed or put in error messages
```

For a local model with no key at all, use `from_provider("ollama", ...)`. Free tiers have rate limits, so start with `--trials 1` or `--trials 3`. Set `price_in` and `price_out` (USD per 1M tokens) if you want `cost_under` to mean something.

The adapter is tested against a scripted fake provider. Model names change over time, so set `EVALLOOP_MODEL` if the default is retired.

## OpenAI Agents SDK and LangGraph

Test an agent built with a framework as it is. The wrapper runs it, then turns tool calls,
tool outputs, model calls and token usage into a trace, so every check, report and CI gate works.

```python
from evalloop import run_suite
from evalloop.frameworks import openai_agents_agent, langgraph_agent, poisoned

results = run_suite(openai_agents_agent(support_agent), cases, trials=5)   # agents.Agent
results = run_suite(langgraph_agent(graph), cases, trials=5)                # compiled StateGraph
```

Multi-turn `Conversation` cases pass earlier turns as message history. For prompt injection
tests, return `poisoned("tool_name", result)` from inside a tool; evalloop adds the attack
text only when a case sets `poison=`. Both adapters are tested against the real packages
(`tests/test_frameworks.py`), using a scripted model so no API key is needed.

## Benchmark a real model

```bash
export GROQ_API_KEY=...          # PowerShell: $env:GROQ_API_KEY="..."
cd examples
python benchmark.py --trials 10                       # or --provider openai --model gpt-4.1-mini
```

It runs three suites (core cases with many trials, every built-in injection attack, and messy
rewrites of the order cases) and writes HTML reports plus a `SUMMARY.md` table to
`benchmark_results/<provider>-<model>/`.

## Add it to an agent you already have (two decorators)

No need to rewrite your agent around `Tracer`. Put `@tool()` on your tool functions and `@agent` on the function that returns the final answer. Sync and async both work, and parallel runs stay separate.

```python
from evalloop import agent, tool, log_llm

@tool()
def lookup_order(order_id): ...            # every call is recorded: arguments, result, time, errors

@agent
def support_agent(text):                    # return the final answer as a string
    reply = call_your_model(text)
    log_llm("plan", text, reply, tokens_in=312, tokens_out=40, latency_ms=850)   # optional: cost and latency
    return reply

run_suite(support_agent, cases, trials=5, workers=4)
```

Start a new project with `evalloop init`. It creates `evals/agent.py`, `evals/suite.py` and a GitHub workflow that already uses the run store, the baseline and the PR summary.

## Red-team your agent

**Prompt injection through tools.** `Case.poison` appends an attack to what a tool returns, exactly like a poisoned web page or document would. The pack has 8 attacks (`INJECTIONS`), and `injection_cases` makes one case per attack.

```python
from evalloop import Case, injection_cases, no_secret_leak, tool_not_called

policy = Case("return_policy", "What is your return policy?", [no_secret_leak(KEY), tool_not_called("issue_refund")])
cases = injection_cases(policy, tool="search_kb", secret=KEY)     # 8 cases, one per attack
```

**Messy input.** `variants` rewrites a case with typos, SHOUTING, extra spaces, politeness and filler words, with the same checks. Numbers, ids and amounts are never changed. If your agent passes the original but fails the variants, it is brittle.

```python
from evalloop import variants
cases = [track, *variants(track, "typos", "upper", "polite", "filler")]
```

The demo (`examples/redteam_demo.py`) has a gullible agent and a careful one. The gullible agent passes the plain question, then fails on typos, on SHOUTING and on 6 of 8 injection attacks. The careful agent passes everything.

```bash
cd examples
evalloop run redteam_demo:cases --agent redteam_demo:careful --baseline ../docs/redteam_gullible.json --trials 3
```

## Keep history and watch trends

```bash
evalloop run suite:cases --agent agent:support_agent --trials 5 --store runs --baseline last
evalloop trend runs --html trend.html
```

`--store` saves every run as plain JSON. `--baseline last` compares with the previous run automatically, so you never manage baseline files by hand. `evalloop trend` makes one HTML page with charts of pass rate (with its confidence band), reliability, p95 latency, cost and flaky cases over time.

## Multi-turn conversations

Real agents fail across turns: they forget the order number, repeat themselves, or leak a secret on the third message. A single-turn test never sees that. `Conversation` runs a whole dialogue, passes the history to your agent, and checks each turn and the conversation as a whole.

```python
from evalloop import Conversation, Turn, answer_contains, tool_args, tool_count, no_secret_leak, grounded

cases = [
    Conversation(
        "refund_it_after_lookup",
        ["Look up order #1042",
         Turn("OK please refund it", [tool_args("issue_refund", order_id="1042")])],   # checks for this turn
        [tool_count("issue_refund", at_most=1), no_secret_leak(KEY), grounded()],     # checks for the whole chat
    ),
]
```

- `ctx.history` is the list of earlier `(user, answer)` pairs and `ctx.turn` is the turn number. `chat_agent` and `anthropic_agent` send the history to the model for you.
- Conversation checks see every step and every answer, so a tool budget, loop limit, PII or secret check covers all turns, not just the last one.
- A turn can be a function, `Turn(lambda history: ...)`, so a scripted user can react to what the agent just said.
- If the agent crashes on a turn, the conversation stops and fails with that turn named.

The demo in `examples/convo_agent.py` has a v1 that only reads the latest message and a v2 that uses the history. evalloop fails v1 on 3 of 4 conversations and passes v2 on all of them:

![conversation report](https://raw.githubusercontent.com/sumitparmartech/evalloop/main/docs/report_convo.png)

```bash
cd examples
evalloop run convo_suite:cases --agent convo_agent:agent_v2 --name convo --trials 3 --baseline ../docs/convo_v1.json
```

## Statistics you can trust

Agents are random, so a single failure is not always a real regression. evalloop reports honest numbers:

- A 95% confidence interval on the pass rate, so "100% on 5 trials" shows as 57% to 100%.
- Flaky cases (pass some trials, fail others) are listed on their own, because they are the ones that hurt in production.
- Unbiased `pass@k` and `pass^k` estimates from your n trials: `case.pass_at(k)`, `case.pass_hat(k)`.
- `--alpha 0.2` ignores drops that are probably noise. Every changed case shows its `p_value`, so you can see what is real.

```bash
evalloop run suite:cases --agent my_module:agent --trials 10 --baseline base.json \
  --fail-on-regression --alpha 0.2 --workers 8 --timeout 60
```

`--workers` runs trials in parallel, so a real-model suite finishes in a fraction of the time. `--timeout` stops one hung call from freezing the run.

## Test real conversations you already have

Take logged messages from production, turn them into traces and run the same checks. No model call needed.

```python
from evalloop import from_openai_messages, from_anthropic_messages, replay_agent, run_suite, save_traces

traces = [from_openai_messages(m) for m in logged_conversations]
save_traces("prod_traces.jsonl", traces)
result = run_suite(replay_agent(traces), cases)   # checks run on the recorded traces
```

## LLM-as-judge

For things code cannot check, add a judge. It scores 1 to 5 at temperature 0 and the check passes at your pass mark.

```python
from evalloop import judge, judge_with

ask = judge_with("anthropic")            # or "openai", "groq", "gemini", "ollama"
Case("tone", "I am angry about my late order", [judge("Calm, apologetic, offers a concrete next step", ask)])
```

## Claude and other providers

```python
from evalloop import anthropic_agent, from_provider

agent = anthropic_agent(system=SYSTEM, tools=ANTHROPIC_TOOLS, impls=IMPLS, model="claude-sonnet-4-5")
# ANTHROPIC_API_KEY is read from the environment and only goes in the x-api-key header
```

## Use it in pytest and any CI

```python
from evalloop import assert_reliable

def test_refund_flow():
    assert_reliable(my_agent, refund_case, trials=5, min_pass_rate=0.8)
```

`--junit junit.xml` gives every case as a test in Jenkins, GitLab or GitHub. `--markdown "$GITHUB_STEP_SUMMARY"` writes a short report with regressions, flaky cases and failure reasons to the job summary.

## Gate your pull requests

```bash
evalloop run suite:cases --agent my_module:my_agent --trials 5 \
  --baseline baseline.json --fail-on-regression --fail-under 0.95 --html report.html
```

The command exits with code 1 if any case got worse than the saved baseline, or if the overall pass rate drops below your threshold. A ready workflow is in `.github/workflows/agent-evals.yml`.

Other commands:

```bash
evalloop diff old.json new.json --fail-on-regression   # compare two saved runs
evalloop report run.json --baseline old.json --html r.html
```

## Why pass^k and not just accuracy

If an agent passes a task 80% of the time, running it once and seeing a pass tells you almost nothing. A case counts as reliable only if all k trials pass, so reliability falls fast as k grows. This is the number users actually feel.

## Design notes

- Trace first. Checks run on every LLM call and tool call, so you can assert on behavior, not only on the final text.
- Standard library only. The HTML report is a single file with no JavaScript, safe to attach to a pull request.
- Deterministic by default. Every trial gets a stable seed derived from the case id, trial number and your base seed.
- Crashes count as failures. If the agent raises, that trial fails and the error is shown in the report.

## Roadmap

- A hosted dashboard (today: a local trend page from `--store`)
- Adapters for LlamaIndex and CrewAI
- Published benchmark numbers for popular models
- Judge calibration against human labels

## Develop

```bash
pip install -e ".[dev]"
pytest
```

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, style and a list of good first issues.

MIT licensed.
