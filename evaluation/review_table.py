"""Export a two-reviewer annotation table and an agreement-stats script.

The table contains only the context a reviewer needs to judge a diagnostic
unit: request, context, evidence needed, evidence seen, model decision, and
reference answer. It HIDES the method name, the machine classification, and
(the other) reviewer's annotation.

No human annotation has been performed; this only sets up the workflow, so no
inter-rater agreement / Cohen's kappa is claimed anywhere.
"""
from __future__ import annotations

import csv
import json

ROOT = "/path/to/habitdrive_eval"
OUT = f"{ROOT}/data/outputs"
DATA = "/path/to/habitdrive_full/data"


def load_tasks():
    route = json.load(open(f"{DATA}/route_tasks.json"))["tasks"]
    cabin = json.load(open(f"{DATA}/cabin_tasks.json"))["tasks"]
    t = {}
    for x in route + cabin:
        t[x["task_id"]] = x
    return t


def build_rows():
    att = json.load(open(f"{OUT}/attribution.json", encoding="utf-8"))
    tasks = load_tasks()
    methods_outputs = {}
    for m in att["units"]:
        methods_outputs[m] = json.load(open(f"{OUT}/{m}.json", encoding="utf-8"))

    rows = []
    for method, units in att["units"].items():
        mout = {t["task_id"]: t for t in methods_outputs[method]["tasks"]}
        for u in units:
            task = tasks.get(u["task_id"])
            if not task:
                continue
            pub = task["public"]
            ev = task["evaluation"]
            rec = mout.get(u["task_id"], {})
            stage = u["stage"]
            decision = rec.get("stages", {}).get(stage, {}).get("parsed", "")
            if u["type"] == "route":
                ref = {"destination": ev["gold"]["destination"]["name"],
                       "route_id": ev["gold"]["route_id"],
                       "parking_id": ev["gold"]["parking_id"]}
            else:
                ref = ev["target_fields"]
            rows.append({
                "review_id": f"{u['task_id']}:{stage}",
                "task_id": u["task_id"],
                "type": u["type"],
                "stage": stage,
                "request": pub["request"]["text"],
                "context": json.dumps(pub.get("context") or
                                      pub.get("environment", {}).get("context", {}),
                                      ensure_ascii=False),
                "evidence_needed": u.get("needed", ""),
                "evidence_seen": u.get("seen", ""),
                "model_decision": json.dumps(decision, ensure_ascii=False),
                "reference_answer": json.dumps(ref, ensure_ascii=False),
            })
    return rows


def export():
    rows = build_rows()
    path = f"{ROOT}/reports/review_table.csv"
    cols = ["review_id", "task_id", "type", "stage", "request", "context",
            "evidence_needed", "evidence_seen", "model_decision",
            "reference_answer", "reviewer_judgement"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "reviewer_judgement": ""})
    print(f"wrote {path} with {len(rows)} rows")


if __name__ == "__main__":
    export()
