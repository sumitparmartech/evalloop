"""Build docs/index.html from docs/site_template.html and the real evalloop runs.

    python docs/build_site.py

The landing page shows the numbers from docs/run_v1.json and docs/run_v2.json,
so regenerating the runs regenerates the site.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def short(value, n=90):
    s = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return s if len(s) <= n else s[: n - 3] + "..."


def pack(path):
    run = json.load(open(os.path.join(HERE, path), encoding="utf-8"))
    s = run["summary"]
    cases = []
    for c in run["cases"]:
        trials = [t["passed"] for t in c["trials"]]
        pick = next((t for t in c["trials"] if not t["passed"]), c["trials"][0])
        cases.append(
            {
                "id": c["id"],
                "input": c["input"],
                "tags": c["tags"],
                "trials": trials,
                "example": {
                    "trial": pick["trial"] + 1,
                    "passed": pick["passed"],
                    "checks": [{"name": k["name"], "passed": k["passed"], "detail": k["detail"]} for k in pick["checks"]],
                    "answer": pick["trace"]["answer"],
                    "steps": [
                        {
                            "kind": st["kind"],
                            "name": st["name"],
                            "ms": round(st["latency_ms"]),
                            "out": short(st["output"], 70),
                        }
                        for st in pick["trace"]["steps"]
                    ],
                },
            }
        )
    return {
        "label": run["label"],
        "suite": run["suite"],
        "trials": run["trials_per_case"],
        "summary": {
            "pass_rate": s["pass_rate"],
            "reliability": s["reliability"],
            "p95": s["latency_p95_ms"],
            "cost": s["cost_per_run_usd"],
            "steps": s["steps_per_run"],
        },
        "cases": cases,
    }


def main():
    data = {"v1": pack("run_v1.json"), "v2": pack("run_v2.json")}
    tpl = open(os.path.join(HERE, "site_template.html"), encoding="utf-8").read()
    eng = open(os.path.join(HERE, "live_engine.js"), encoding="utf-8").read().replace("</", "<\\/")
    out = tpl.replace("/*__ENGINE__*/", eng).replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    with open(os.path.join(HERE, "index.html"), "w", encoding="utf-8") as f:
        f.write(out)
    print("wrote docs/index.html", len(out), "bytes")


if __name__ == "__main__":
    main()
