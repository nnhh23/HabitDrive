"""Finalize the smoke outputs after cancelling the representative-method
tail: write representative outputs as `blocked` (no scores) and rebuild
summary.json from the saved basic-method results."""
from __future__ import annotations

import json

ROOT = "/path/to/habitdrive_eval"
OUT = f"{ROOT}/data/outputs"

BASIC = ["no_memory", "oracle_memory", "full_context",
         "recursive_summarization", "vector_rag", "key_value_store"]
REP = {"mem0": "installed", "lightmem": "installed",
       "memobase": "installed", "memoryos": "installed",
       "a_mem": "not_installed"}


def main():
    manifest = json.load(open(f"{ROOT}/data/smoke/manifest.json"))
    rep_status = json.load(open(f"{ROOT}/reports/representative_status.json"))

    summary = []
    for m in BASIC:
        d = json.load(open(f"{OUT}/{m}.json", encoding="utf-8"))
        tasks = d["tasks"]
        summary.append({
            "method": m, "status": "e2e_run", "error": None,
            "n_tasks": len(tasks),
            "n_exceptions": sum(1 for t in tasks if t["status"] != "executed"),
            "n_parse_fail": sum(1 for t in tasks if t.get("parse_failure")),
            "metrics": d["metrics"], "elapsed_s": None,
        })

    for m, st in REP.items():
        r = rep_status[m]
        # write a blocked output file (no task scores)
        out = {
            "method": m, "status": "blocked",
            "error": r["blocked_reason"],
            "model": "n/a (blocked)",
            "metrics": None,
            "tasks": [],
        }
        with open(f"{OUT}/{m}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        summary.append({
            "method": m, "status": "blocked",
            "error": r["blocked_reason"],
            "n_tasks": 0, "n_exceptions": 0, "n_parse_fail": 0,
            "metrics": None, "elapsed_s": None,
        })

    with open(f"{OUT}/summary.json", "w", encoding="utf-8") as f:
        json.dump({"manifest": manifest,
                   "model": "Llama-3.1-8B-Instruct (substitute)",
                   "summary": summary}, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
