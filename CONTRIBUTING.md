# Contributing to evalloop

Thanks for helping. Bug reports, new checks, adapters and docs fixes are all welcome.

## Set up

```bash
git clone https://github.com/sumitparmartech/evalloop
cd evalloop
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

The framework adapter tests run only when the frameworks are installed:

```bash
pip install -e ".[dev,agents,langgraph]"
pytest tests/test_frameworks.py
```

## Ground rules

- **No new runtime dependencies.** The core stays standard library only. Framework code
  imports its package inside the function that needs it.
- **Every change has a test.** Tests must not call a real model or the network. Use a
  scripted fake (see `tests/test_adapters.py` and `tests/test_frameworks.py`).
- **Checks look at the trace.** A new check gets the `Trace` and returns a `CheckResult`
  with a short, specific reason when it fails.
- **Keep reports readable.** If you add data to the HTML report, check it in light and
  dark mode.
- **Never log secrets.** API keys come from environment variables and must not appear in
  errors, reports or traces.

## Pull requests

1. Open an issue first for anything bigger than a small fix, so we can agree on the shape.
2. Keep the PR focused on one change.
3. Run `pytest` and add a line to `CHANGELOG.md` under "Unreleased".

## Good first issues

- A `no_refusal()` check that fails when the agent refuses a valid request.
- A `--only TAG` flag for `evalloop run` to run cases with one tag.
- A LlamaIndex adapter in `evalloop/frameworks.py` (same pattern as LangGraph).
- A CrewAI adapter.
- An `evalloop compare a.json b.json c.json` command that prints a table across models.
- Show the per-case confidence interval in the HTML report.
- More messy-input variants (for example, other languages or voice-to-text style errors).

## Reporting a security issue

Please do not open a public issue. Email the maintainer listed on the GitHub profile.
