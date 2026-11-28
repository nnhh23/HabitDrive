"""Error attribution (paper Figure 4): for each failed diagnostic unit,
compare the evidence NEEDED for a correct decision against the evidence the
model actually SAW, and classify memory-acquisition vs decision error
(undeterminable when the record is missing/incomplete).

Diagnostic units:
  - route: the first failed stage (destination / route / parking).
  - cabin: each incorrect target field (or, if targets all correct, the
    protected/constraint violation).

Reference ("needed") evidence is taken from the Oracle run (valid, task-
relevant preference facts); "seen" evidence is the memory text each method
actually supplied at that stage (recorded in the smoke outputs).
"""
from __future__ import annotations

import json
import re

ROOT = "/path/to/habitdrive_eval"
OUT = f"{ROOT}/data/outputs"


def key_tokens(text):
    toks = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) > 2:
            toks.add(w)
    return toks


def needed_facts_for(oracle_tasks, task_id):
    for t in oracle_tasks:
        if t["task_id"] == task_id:
            return t
    return None


def stage_needed_seen(unit_stage, oracle_task):
    """Return (needed_text, seen_text) for a stage."""
    if not oracle_task:
        return "", ""
    stage = oracle_task["stages"].get(unit_stage, {})
    return stage.get("memory", ""), ""


def classify(needed_text, seen_text):
    nt = key_tokens(needed_text)
    st = key_tokens(seen_text)
    if not nt:
        return "undeterminable"
    overlap = len(nt & st)
    ratio = overlap / len(nt)
    if ratio >= 0.5:
        return "decision_error"
    return "memory_acquisition_error"


def attribute(method_name):
    oracle = json.load(open(f"{OUT}/oracle_memory.json", encoding="utf-8"))
    oracle_by_id = {t["task_id"]: t for t in oracle["tasks"]}
    m = json.load(open(f"{OUT}/{method_name}.json", encoding="utf-8"))
    units = []
    for t in m["tasks"]:
        if t["status"] != "executed":
            units.append({"task_id": t["task_id"], "type": t["type"],
                          "stage": "execution", "needed": "", "seen": "",
                          "classification": "undeterminable",
                          "reason": t.get("error", "missing record")})
            continue
        if t["type"] == "route":
            if t["score"].get("e2e"):
                continue
            # first failed stage
            reason = t.get("stop_reason", "unknown")
            stage = {"destination_wrong": "destination",
                     "route_wrong": "route",
                     "parking_wrong": "parking"}.get(reason, "destination")
            ot = oracle_by_id.get(t["task_id"])
            needed = ot["stages"].get(stage, {}).get("memory", "") if ot else ""
            seen = t["stages"].get(stage, {}).get("memory", "")
            units.append({"task_id": t["task_id"], "type": "route", "stage": stage,
                          "needed": needed, "seen": seen,
                          "classification": classify(needed, seen),
                          "reason": reason})
        else:  # cabin
            if t["score"].get("e2e"):
                continue
            s = t["score"]
            incorrect = [f for f, ok in s.get("field_correct", {}).items() if not ok]
            ot = oracle_by_id.get(t["task_id"])
            if incorrect:
                for f in incorrect:
                    needed = ot["stages"].get("cabin", {}).get("memory", "") if ot else ""
                    seen = t["stages"].get("cabin", {}).get("memory", "")
                    units.append({"task_id": t["task_id"], "type": "cabin",
                                  "stage": f"field:{f}", "needed": needed,
                                  "seen": seen, "classification": classify(needed, seen),
                                  "reason": "incorrect_target_field"})
            else:
                # all targets correct but protected/constraint violated
                v = s.get("protected_violated", []) + s.get("constraint_violated", [])
                for item in v:
                    units.append({"task_id": t["task_id"], "type": "cabin",
                                  "stage": f"violation:{item}", "needed": "",
                                  "seen": t["stages"].get("cabin", {}).get("memory", ""),
                                  "classification": "decision_error",
                                  "reason": "protected_or_constraint_violation"})
    return units


def summarize(method_name, units):
    from collections import Counter
    c = Counter(u["classification"] for u in units)
    total = len(units)
    dec = c.get("decision_error", 0)
    mem = c.get("memory_acquisition_error", 0)
    und = c.get("undeterminable", 0)
    classified = dec + mem
    return {
        "method": method_name, "total_units": total, "classified": classified,
        "excluded_undeterminable": und,
        "memory_acquisition_errors": mem, "decision_errors": dec,
        "mem_share": (mem / classified * 100) if classified else None,
        "dec_share": (dec / classified * 100) if classified else None,
    }


if __name__ == "__main__":
    import sys
    methods = [m for m in ["no_memory", "full_context",
                           "recursive_summarization", "vector_rag",
                           "key_value_store"]]
    summary = []
    all_units = {}
    for m in methods:
        units = attribute(m)
        all_units[m] = units
        summary.append(summarize(m, units))
    with open(f"{OUT}/attribution.json", "w") as f:
        json.dump({"summary": summary, "units": all_units}, f, indent=1,
                  ensure_ascii=False)
    print(json.dumps(summary, indent=1))
