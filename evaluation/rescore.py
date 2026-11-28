"""Re-score from saved outputs only (no model calls).

Reads data/outputs/<method>.json and recomputes route/cabin metrics from the
saved per-task `score` fields, so results can be re-aggregated without
re-running the decision model.
"""
from __future__ import annotations

import json

import scoring

ROOT = "/path/to/habitdrive_eval"
OUT = f"{ROOT}/data/outputs"
BASIC = ["no_memory", "oracle_memory", "full_context",
         "recursive_summarization", "vector_rag", "key_value_store"]


def main():
    print(f"{'method':26s} route E2E   DA     PSA   | cabin E2E  TFA")
    for m in BASIC:
        d = json.load(open(f"{OUT}/{m}.json", encoding="utf-8"))
        r_pts = [t["score"] for t in d["tasks"] if t["type"] == "route"
                 and "destination_correct" in t["score"]]
        c_pts = [t["score"] for t in d["tasks"] if t["type"] == "cabin"
                 and "field_correct" in t["score"]]
        r = scoring.route_metrics(r_pts) if r_pts else {}
        c = scoring.cabin_metrics(c_pts) if c_pts else {}

        def p(x):
            return f"{x*100:.1f}" if x is not None else "n/a"
        print(f"{m:26s} {p(r.get('E2E')):>8s} {p(r.get('DA')):>6s} "
              f"{p(r.get('PSA')):>6s} | {p(c.get('E2E')):>8s} {p(c.get('TFA')):>5s}")


if __name__ == "__main__":
    main()
