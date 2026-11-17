"""`evalloop init`: a working starter project that uses the decorators."""
from __future__ import annotations

import os

AGENT = '''"""Replace the body of support_agent with your own agent. Keep the two decorators."""
from evalloop import agent, tool, log_llm

ORDERS = {"1042": {"status": "delivered", "total": 49.0}, "1043": {"status": "in_transit", "total": 129.5}}


@tool()
def lookup_order(order_id: str) -> dict:
    return ORDERS.get(order_id, {"error": "not_found"})


@agent
def support_agent(text: str) -> str:
    # Call your model here and log it so cost and latency show up in the report:
    #   reply = client.chat(...);  log_llm("plan", prompt, reply, tokens_in=..., tokens_out=...)
    order_id = "".join(c for c in text if c.isdigit())[:4]
    order = lookup_order(order_id)
    if "error" in order:
        return f"I could not find order {order_id}."
    return f"Order {order_id} is {order['status'].replace('_', ' ')} and the total was ${order['total']:.2f}."
'''

SUITE = '''from evalloop import Case, answer_contains, grounded, max_steps, no_loop, no_pii, tool_args, variants

base = Case("track_order", "Where is my order #1043?",
            [tool_args("lookup_order", order_id="1043"), answer_contains("in transit"), grounded(), no_loop(2), max_steps(6)])

cases = [
    base,
    Case("unknown_order", "Where is my order #9999?", [answer_contains("could not find"), grounded(), no_pii()]),
    # the same case with typos, SHOUTING and filler words: is your agent robust to how people really write?
    *variants(base, "typos", "upper", "filler"),
]
'''

WORKFLOW = '''name: agent-evals
on: [pull_request]
jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install evalloop
      - uses: actions/cache@v4
        with: { path: evals/runs, key: evalloop-runs-${{ github.run_id }}, restore-keys: evalloop-runs- }
      - name: Run evals
        working-directory: evals
        run: |
          evalloop run suite:cases --agent agent:support_agent --trials 5 --workers 4 \\
            --store runs --baseline last --alpha 0.2 --fail-on-regression --fail-under 0.9 \\
            --html report.html --junit junit.xml --markdown "$GITHUB_STEP_SUMMARY"
      - uses: actions/upload-artifact@v4
        if: always()
        with: { name: evalloop-report, path: "evals/report.html\\nevals/junit.xml" }
'''


def scaffold(folder: str) -> list:
    files = {
        os.path.join(folder, "agent.py"): AGENT,
        os.path.join(folder, "suite.py"): SUITE,
        os.path.join(".github", "workflows", "agent-evals.yml"): WORKFLOW,
    }
    made = []
    for path, text in files.items():
        if os.path.exists(path):
            print("skipped (exists):", path)
            continue
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        made.append(path)
    return made
