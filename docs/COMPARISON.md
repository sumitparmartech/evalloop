# evalloop and other eval tools

Short, honest notes on where evalloop fits. Tools change fast, so check their docs
before you decide. Last checked: November 2026.

## What evalloop focuses on

evalloop is a small Python library for one job: **regression tests for tool-using agents
that run in CI**.

- **Checks on the full trace, not only the answer.** Which tools ran, in what order, with
  what arguments, whether the answer is grounded in tool output, loops, step budgets,
  cost and latency.
- **Reliability as a first-class number.** Every case runs many times. The report shows
  pass rate with a 95% confidence interval and pass^k (the share of cases that passed
  every trial), and marks flaky cases.
- **Noise-aware diffs.** Comparing two runs uses a two-proportion test, so a drop that is
  probably luck is labelled as noise instead of failing the build.
- **Attacks and messy input built in.** `Case(poison=...)` injects attack text into a tool
  result; `variants(case)` rewrites inputs with typos, caps and filler.
- **Explains flaky cases.** `evalloop explain` compares passing and failing trials of the same case
  and reports where they split (a skipped tool, a changed argument, a loop), with counts.
- **Side-by-side model comparison.** `evalloop compare` puts runs of the same suite next to each other.
- **No dependencies.** Plain Python, a single-file HTML report, JUnit and Markdown output.
- **Works with what you have.** Decorators for plain Python agents, adapters for any
  OpenAI-compatible API, Anthropic, the OpenAI Agents SDK and LangGraph, and importers for
  logged OpenAI or Anthropic message lists.

## Other tools, briefly

| Tool | Good fit when | Repeated runs |
|---|---|---|
| promptfoo | You want a YAML-driven CLI to compare prompts and providers, plus red-teaming | Yes: `repeat` option and `--repeat` flag |
| DeepEval | You want pytest-style tests with many LLM-judged metrics (RAG, agents, conversations) | Check its docs |
| Inspect AI | You run research-style benchmarks with solvers, scorers and epochs | Yes: epochs with reducers such as `pass_at_k` |
| Pydantic Evals | You already use Pydantic AI and want datasets and evaluators in Python | Yes: `repeat` parameter |
| LangSmith, Phoenix and similar | You want a hosted platform: tracing, datasets, dashboards, human review | Phoenix has repetitions for experiments |

## When to pick evalloop

- Your agent calls tools and you care **what it did**, not just what it said.
- You want a **merge gate** that does not block on random noise.
- You want **no service, no account and no extra packages**.

## When to pick something else

- You need many judge-based quality metrics (faithfulness, relevancy and so on): DeepEval.
- You compare prompts across many providers from config files: promptfoo.
- You need a team dashboard, production tracing and annotation queues: a hosted platform.

These tools can also work together. For example, log traces in a platform, export the
failures, and use `evalloop.from_openai_messages` to turn them into regression cases.
