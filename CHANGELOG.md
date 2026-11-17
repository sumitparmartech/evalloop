# Changelog

## 0.5.1

- New "Signal Light" look for the project site and the HTML report (warm paper background, amber accent, clearer pass and fail colours).
- Site: new "New in 0.5" section (multi-turn, prompt injection, messy input, history and trends), updated CI snippet and decorator example.
- Reports and screenshots regenerated. No API changes.

## 0.5.0

- `@agent` and `@tool` decorators (sync and async) and `log_llm`: test an existing agent without rewriting it. Safe with parallel trials.
- Red team: `Case.poison` appends attack text to a tool result. `INJECTIONS` (8 attacks) and `injection_cases(...)`.
- `variants(case, ...)`: typos, upper, lower, whitespace, polite, filler and terse rewrites that never touch numbers or ids.
- Run history: `--store DIR`, `--baseline last`, and `evalloop trend` for an HTML trend report.
- `evalloop init` scaffolds an example agent, suite and GitHub workflow.
- Demo: `examples/redteam_demo.py`.

## 0.4.0

- Multi-turn conversations: `Conversation(id, turns, checks)` and `Turn(say, checks)`. The agent gets `ctx.history` and `ctx.turn`.
- Per-turn checks and whole-conversation checks. Budgets, loops, `no_secret_leak`, `no_pii` and `grounded` now cover every turn.
- A `Turn` can be a function of the history, for scripted users that react to the agent.
- `chat_agent` and `anthropic_agent` send the earlier turns to the model.
- The HTML report shows each turn of a conversation. Demo: `examples/convo_agent.py` and `examples/convo_suite.py`.

## 0.3.0

- Statistics: 95% Wilson confidence intervals, unbiased `pass@k` and `pass^k` estimators, flaky-case detection.
- `diff_runs(..., alpha=0.2)` and `--alpha`: ignore drops that are likely noise. Every diff row now has `p_value` and `noise`.
- Parallel trials (`workers=` / `--workers`) and per-trial `timeout` (`--timeout`). Results keep their order.
- New checks: `judge` (LLM-as-judge), `no_pii`, `tool_count`. `tool_args` now accepts predicates and regexes.
- Native Anthropic Messages adapter (`anthropic_agent`) and `judge_with` for any provider.
- Import real conversations: `from_openai_messages`, `from_anthropic_messages`, `save_traces`, `load_traces`, `replay_agent`.
- CI: JUnit XML (`--junit`), Markdown summary for PRs (`--markdown`), and `assert_reliable` for pytest.
- Report shows the confidence interval and flaky cases. Package ships `py.typed`.

## 0.2.0

- OpenAI-compatible adapter (`chat_agent`, `from_provider`), HTML report, CLI diff and CI gate.
