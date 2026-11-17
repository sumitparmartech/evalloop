"""A small multi-turn demo agent in two versions (no API key needed).

v1 only reads the latest message, so it forgets the order the user already named.
v2 uses ctx.history. Real agents fail this way all the time, and a single-turn
test never sees it.
"""
import re

from evalloop import Tracer
from support_agent import ORDERS, SECRET


def _order_id(text):
    m = re.search(r"#?(\d{4})", text)
    return m.group(1) if m else None


def _make(version):
    remembers = version == "v2"

    def agent(text, ctx):
        t = Tracer(text)
        low = text.lower()
        oid = _order_id(text)
        if oid is None and remembers:  # look back through the conversation
            for said, _ in reversed(ctx.history):
                oid = _order_id(said)
                if oid:
                    break
        if "ignore" in low and "instructions" in low:
            t.llm("answer", text, "refuse", tokens_in=200, tokens_out=12, latency_ms=300)
            return t.finish("I can't share internal credentials.")
        if oid is None:
            t.llm("answer", text, "ask", tokens_in=200, tokens_out=12, latency_ms=300)
            return t.finish("Which order do you mean? Please send the order number.")
        t.llm("plan", text, f"lookup {oid}", tokens_in=200, tokens_out=14, latency_ms=300)
        order = t.tool("lookup_order", {"order_id": oid}, output=ORDERS.get(oid, {"error": "not_found"}), latency_ms=120)
        if "refund" in low and "error" not in order:
            t.tool("issue_refund", {"order_id": oid, "amount": order["total"]}, output={"ok": True}, latency_ms=120)
            return t.finish(f"Done. I refunded ${order['total']:.2f} for order {oid}.")
        if order.get("status") == "in_transit":
            return t.finish(f"Order {oid} is in transit and should arrive in {order['eta_days']} days.")
        return t.finish(f"Order {oid} is {order.get('status', 'unknown')}.")

    return agent


agent_v1 = _make("v1")
agent_v2 = _make("v2")
