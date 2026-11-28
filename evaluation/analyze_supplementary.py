"""Supplementary analyses (CPU, from saved outputs):
  - household-clustered bootstrap CIs for E2E
  - intermediate memory diagnostics (evidence recall, user attribution,
    update retention) broken down by challenge label
  - per-challenge-group E2E
  - excluded diagnostic-unit counts

Reads data/outputs/<method>.json and data/smoke/manifest.json; no model calls.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict

ROOT = "/path/to/habitdrive_eval"
OUT = f"{ROOT}/data/outputs"
DATA = "/path/to/habitdrive_full/data"

METHODS = ["no_memory", "oracle_memory", "full_context",
           "recursive_summarization", "vector_rag", "key_value_store",
           "recent_history", "full_context_8k"]
LABELS = ["user_binding", "preference_update", "conditional_application",
          "temporary_request_interference"]


def toks(text):
    return set(w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
               if len(w) > 2)


def load_outputs(name):
    try:
        return json.load(open(f"{OUT}/{name}.json", encoding="utf-8"))
    except FileNotFoundError:
        return None


def bootstrap_ci(e2e_by_household, n=2000, seed=42):
    """Cluster bootstrap over households."""
    rng = random.Random(seed)
    hids = list(e2e_by_household.keys())
    if not hids:
        return None
    vals = []
    for _ in range(n):
        s = [rng.choice(hids) for _ in hids]
        xs = [x for h in s for x in e2e_by_household[h]]
        vals.append(sum(xs) / len(xs))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def _owner_present(seen_memory, requester_name, requester_id):
    blob = (seen_memory or "").lower()
    return (requester_name.lower() in blob) or (requester_id.lower() in blob)


def main():
    manifest = json.load(open(f"{ROOT}/data/smoke/manifest.json",
                              encoding="utf-8"))
    task_meta = {t["task_id"]: t for t in manifest["tasks"]}
    # requester identity per task (from full task files)
    requester = {}
    for fn in ("route_tasks", "cabin_tasks"):
        try:
            tasks = json.load(open(f"{DATA}/{fn}.json", encoding="utf-8"))["tasks"]
        except FileNotFoundError:
            continue
        for t in tasks:
            requester[t["task_id"]] = (t["public"]["request"]["speaker_name"],
                                       t["public"]["request"]["speaker_id"])
    oracle = load_outputs("oracle_memory")
    oracle_by_task = {}
    if oracle:
        for t in oracle["tasks"]:
            oracle_by_task[t["task_id"]] = t

    rows = {}
    for m in METHODS:
        d = load_outputs(m)
        if d is None:
            continue
        # ---- per-task E2E + memory (needed/seen) ----
        route_e2e = defaultdict(list)   # hid -> [0/1]
        cabin_e2e = defaultdict(list)
        group_e2e = defaultdict(list)   # label -> [0/1]
        ev_recall = defaultdict(list)   # label -> recall values (and "all")
        ua = defaultdict(list)
        seen_len = []
        for t in d["tasks"]:
            hid = t["household_id"]
            mt = task_meta.get(t["task_id"], {})
            labs = mt.get("challenge_labels", [])
            sc = t.get("score", {})
            if t["type"] == "route" and "e2e" in sc:
                route_e2e[hid].append(1 if sc["e2e"] else 0)
                for lab in labs:
                    group_e2e[lab].append(1 if sc["e2e"] else 0)
            elif t["type"] == "cabin" and "e2e" in sc:
                cabin_e2e[hid].append(1 if sc["e2e"] else 0)
                for lab in labs:
                    group_e2e[lab].append(1 if sc["e2e"] else 0)
            # evidence recall vs oracle (needed) at each stage
            ot = oracle_by_task.get(t["task_id"])
            if ot:
                for stage, st in t.get("stages", {}).items():
                    ost = ot.get("stages", {}).get(stage, {})
                    needed = toks(ost.get("memory", ""))
                    seen = toks(st.get("memory", ""))
                    if not needed:
                        continue
                    r = len(needed & seen) / len(needed)
                    ev_recall["all"].append(r)
                    for lab in labs:
                        ev_recall[lab].append(r)
                    # user attribution: correct speaker name/id present in seen
                    if "user_binding" in labs:
                        rn, rid = requester.get(t["task_id"], ("", ""))
                        ua["user_binding"].append(
                            1 if _owner_present(st.get("memory", ""), rn, rid)
                            else 0)
                    seen_len.append(len(seen))

        route_e2e_all = [x for v in route_e2e.values() for x in v]
        cabin_e2e_all = [x for v in cabin_e2e.values() for x in v]

        def mean(xs):
            return round(sum(xs) / len(xs), 4) if xs else None

        rows[m] = {
            "route": {
                "n": len(route_e2e_all), "e2e": mean(route_e2e_all),
                "ci95": bootstrap_ci(route_e2e),
            },
            "cabin": {
                "n": len(cabin_e2e_all), "e2e": mean(cabin_e2e_all),
                "ci95": bootstrap_ci(cabin_e2e),
            },
            "group_e2e": {lab: {"n": len(group_e2e[lab]),
                                "e2e": mean(group_e2e[lab])}
                          for lab in LABELS if group_e2e[lab]},
            "evidence_recall": {lab: mean(ev_recall[lab])
                                for lab in ["all"] + LABELS
                                if ev_recall[lab]},
            "user_attribution": mean(ua["user_binding"]),
            "seen_vocab_mean": mean(seen_len) if seen_len else None,
        }

    # excluded counts from attribution.json
    excluded = {}
    try:
        att = json.load(open(f"{OUT}/attribution.json", encoding="utf-8"))
        for s in att.get("summary", []):
            excluded[s["method"]] = {
                "total_units": s.get("total_units"),
                "classified": s.get("classified"),
                "excluded_undeterminable": s.get("excluded_undeterminable"),
            }
    except FileNotFoundError:
        pass

    result = {"methods": rows, "excluded_units": excluded}
    with open(f"{ROOT}/reports/supplementary_analysis.json", "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)

    # pretty print
    print("=== E2E with household-clustered bootstrap 95% CI ===")
    for m, r in rows.items():
        rc = r["route"]; cc = r["cabin"]
        def ci(v):
            return f"[{v[0]:.3f},{v[1]:.3f}]" if v else "n/a"
        print(f"{m:26s} route E2E={_p(rc['e2e'])} CI95={ci(rc['ci95'])} | "
              f"cabin E2E={_p(cc['e2e'])} CI95={ci(cc['ci95'])}")
    print("\n=== evidence recall (vs Oracle) by label ===")
    for m, r in rows.items():
        er = r["evidence_recall"]
        print(f"{m:26s} all={_p2(er.get('all'))} "
              f"update={_p2(er.get('preference_update'))} "
              f"binding={_p2(er.get('user_binding'))} "
              f"cond={_p2(er.get('conditional_application'))} "
              f"tmp={_p2(er.get('temporary_request_interference'))}")
    print("\n=== excluded diagnostic units ===")
    print(json.dumps(excluded, indent=1))


def _p(x):
    return f"{x*100:.1f}%" if x is not None else "n/a"


def _p2(x):
    return f"{x:.3f}" if x is not None else "n/a"


if __name__ == "__main__":
    main()
