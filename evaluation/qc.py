"""Data quality check for the generated HabitDrive dataset.

Reads the final on-disk files and independently checks:
  - parseability, counts (vs paper targets), ID uniqueness,
  - cross-file reference validity, chronology, task-after-history,
  - reference-answer validity (route: recompute requirements & unique gold;
    cabin: target/protected/constraint validity),
  - challenge-label coverage (multi-label counted per label, not summed),
  - natural-language fact preservation on a seeded sample.

Note on independence: reference-answer validity re-uses the generator's
requirement resolver (`resolve_route_requirements` /
`resolve_parking_requirements`) because re-implementing the exact condition
semantics in a second place would risk divergence. Everything else (counts,
IDs, references, chronology, cabin constraints, NL sampling) is independent.

Outputs: reports/qc_report.md and reports/qc_issues.jsonl
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict

GEN = "/path/to/habitdrive_full/code"
DATA = "/path/to/habitdrive_full/data"
ROOT = "/path/to/habitdrive_eval"

sys.path.insert(0, GEN)
import constants as C            # noqa: E402
import generate_structure as gs  # noqa: E402

PAPER = {"households": 100, "base_events": 5000,
         "route_tasks": 2000, "cabin_tasks": 1000}
LABELS = {"user_binding", "preference_update", "conditional_application",
          "temporary_request_interference"}
SAMPLE_SEED = 20251126


def load(name):
    return json.load(open(os.path.join(DATA, name), encoding="utf-8"))


def main():
    issues = []  # (severity, check, sample_id, detail)

    # ---- parseability ----
    try:
        households = load("households.json")["households"]
        memories = load("memories.json")["memories"]
        hist = load("structured_history.json")["histories"]
        nl = load("natural_language_history.json")["histories"]
        route_tasks = load("route_tasks.json")["tasks"]
        cabin_tasks = load("cabin_tasks.json")["tasks"]
    except Exception as e:
        print(f"FATAL parse error: {e}")
        sys.exit(1)

    # ---- counts ----
    base_events = sum(1 for h in hist for e in h["events"]
                      if e["event_type"] in ("establishment", "evolution"))
    expanded = sum(1 for h in hist for e in h["events"]
                   if e["event_type"] not in ("establishment", "evolution"))
    parking = sum(1 for t in route_tasks if t["public"]["parking_required"])
    counts = {
        "households": len(households),
        "members": sum(len(h["members"]) for h in households),
        "memories": len(memories),
        "base_events": base_events,
        "expanded_events": expanded,
        "route_tasks": len(route_tasks),
        "route_tasks_with_parking": parking,
        "cabin_tasks": len(cabin_tasks),
    }
    for k, want in PAPER.items():
        got = counts[k]
        if got != want:
            issues.append(("error", "count", k, f"{got} != {want}"))

    # ---- ID uniqueness ----
    def dupes(seq, key, name):
        c = Counter(key(x) for x in seq)
        return [(k, v) for k, v in c.items() if v > 1][:5]

    for name, seq, key in [
        ("household_id", households, lambda x: x["household_id"]),
        ("member_id", [m for h in households for m in h["members"]],
         lambda x: x["member_id"]),
        ("memory_id", memories, lambda x: x["memory_id"]),
        ("event_id", [e for h in hist for e in h["events"]],
         lambda x: x["event_id"]),
        ("route_task_id", route_tasks, lambda x: x["task_id"]),
        ("cabin_task_id", cabin_tasks, lambda x: x["task_id"]),
    ]:
        d = dupes(seq, key, name)
        if d:
            issues.append(("error", "duplicate_id", name, str(d)))
        # also check for duplicates within whole set (list may be empty)
        if not seq:
            issues.append(("error", "empty", name, "empty sequence"))

    # ---- cross-file references ----
    hid_set = {h["household_id"] for h in households}
    mem_ids = {m["memory_id"]: m for m in memories}
    member_ids = {m["member_id"] for h in households for m in h["members"]}
    mem_by_household = defaultdict(list)
    for m in memories:
        mem_by_household[m["household_id"]].append(m)

    for m in memories:
        if m["household_id"] not in hid_set:
            issues.append(("error", "reference", m["memory_id"], "bad household"))
        if m["owner_id"] not in member_ids:
            issues.append(("error", "reference", m["memory_id"], "bad owner"))

    # history: speaker valid, chronological, event id unique
    for h in hist:
        ts = [e["timestamp"] for e in h["events"]]
        if ts != sorted(ts):
            issues.append(("error", "chronology", h["household_id"],
                           "history not chronological"))
        for e in h["events"]:
            if e["speaker_id"] not in member_ids:
                issues.append(("error", "reference", e["event_id"],
                               "bad speaker"))

    # ---- task-after-history ----
    hist_end = {h["household_id"]: max(e["timestamp"] for e in h["events"])
                for h in hist}
    for t in route_tasks + cabin_tasks:
        if t["household_id"] not in hid_set:
            issues.append(("error", "reference", t["task_id"], "bad household"))
        if t["household_id"] not in hist_end:
            issues.append(("error", "reference", t["task_id"],
                           "household has no history"))
        else:
            ts = t.get("timestamp", "")
            if ts and ts <= hist_end[t["household_id"]]:
                issues.append(("error", "chronology", t["task_id"],
                               "task before/at history end"))

    # ---- challenge labels ----
    label_counter = Counter()
    for t in route_tasks + cabin_tasks:
        for lab in t["evaluation"]["challenge_labels"]:
            if lab not in LABELS:
                issues.append(("error", "label", t["task_id"], f"bad label {lab}"))
            label_counter[lab] += 1

    # ---- reference-answer validity ----
    active = gs.final_active(memories)
    route_gold_mismatch = 0
    route_unique_fail = 0
    cabin_target_bad = 0
    cabin_constraint_bad = 0
    for t in route_tasks:
        pub, ev = t["public"], t["evaluation"]
        requester = pub["request"]["speaker_id"]
        ctx = pub["context"]
        role = ev["gold"]["destination"]["semantic_role"]
        occ = ctx["occupant_ids"]
        reqs, _ = gs.resolve_route_requirements(active, requester, role, ctx, occ)
        if not reqs:
            reqs = [(None, "travel_time", "minimize")]
        routes = pub["stages"]["route_selection"]["candidates"]
        sat = [r for r in routes if all(gs.route_attr_ok(r, a, v)
                                        for (_, a, v) in reqs)]
        gold = ev["gold"]["route_id"]
        if len(sat) != 1:
            route_unique_fail += 1
            issues.append(("error", "route_unique_gold", t["task_id"],
                           f"{len(sat)} routes satisfy"))
        elif sat[0]["route_id"] != gold:
            route_gold_mismatch += 1
            issues.append(("error", "route_gold", t["task_id"],
                           "gold not unique satisfying route"))
        if pub["parking_required"]:
            preqs, _ = gs.resolve_parking_requirements(active, requester, role,
                                                       ctx, occ)
            lots = pub["stages"]["parking_selection"]["candidates"]
            psat = [p for p in lots if all(gs.parking_attr_ok(p, a, v)
                                           for (_, a, v) in preqs)]
            goldp = ev["gold"]["parking_id"]
            if len(psat) != 1:
                issues.append(("error", "parking_unique_gold", t["task_id"],
                               f"{len(psat)} parking satisfy"))
            elif psat[0]["parking_id"] != goldp:
                issues.append(("error", "parking_gold", t["task_id"],
                               "gold parking not unique"))
        # gold destination memory valid
        dmem = ev["gold"]["destination"]["memory_id"]
        if dmem not in mem_ids:
            issues.append(("error", "route_destination", t["task_id"],
                           "gold destination memory missing"))

    all_fields = set(gs.all_cabin_fields())
    for t in cabin_tasks:
        ev, pub = t["evaluation"], t["public"]
        init = pub["environment"]["initial_device_state"]
        for f in ev["target_fields"]:
            if f not in all_fields:
                cabin_target_bad += 1
                issues.append(("error", "cabin_target_field", t["task_id"], f))
        # protected = all - target
        if set(ev["protected_fields"]) != (all_fields - set(ev["target_fields"])):
            issues.append(("error", "cabin_protected", t["task_id"],
                           "protected != all - target"))
        # constraints: no mutual exclusion violated in gold final state
        final = dict(init)
        final.update(ev["target_fields"])
        for a, b in C.CABIN_MUTUAL_EXCLUSION:
            for zone in ["driver", "front_passenger", "rear"]:
                va = final.get(f"{zone}.{a}", "off")
                vb = final.get(f"{zone}.{b}", "off")
                if va != "off" and vb != "off":
                    cabin_constraint_bad += 1
                    issues.append(("error", "cabin_constraint", t["task_id"],
                                   f"{zone} {a}={va} & {b}={vb} both active"))

    # ---- NL fact preservation sample (programmatic) ----
    rng = random.Random(SAMPLE_SEED)
    sampled = rng.sample(hist, min(20, len(hist)))
    nl_by_hid = {h["household_id"]: h for h in nl}
    nl_fact_fail = 0
    nl_samples = []
    for h in sampled:
        hid = h["household_id"]
        nlg = nl_by_hid.get(hid)
        if not nlg:
            continue
        turns = {m["event_id"]: m for m in nlg["history"] if m["role"] == "user"}
        for e in h["events"]:
            if e["event_type"] == "unrelated":
                continue
            u = turns.get(e["event_id"])
            if u and e.get("fact") and not gs_fact_preserved(e["fact"],
                                                             u["text"]):
                nl_fact_fail += 1
                nl_samples.append((hid, e["event_id"]))
        # tasks: sample a couple requests
    # sample task requests for review list
    rt_samp = rng.sample(route_tasks, min(10, len(route_tasks)))
    ct_samp = rng.sample(cabin_tasks, min(10, len(cabin_tasks)))

    # ---- write report ----
    report = {
        "counts": counts,
        "label_counts": dict(label_counter),
        "paper_targets": PAPER,
        "checks": {
            "route_gold_mismatch": route_gold_mismatch,
            "route_unique_fail": route_unique_fail,
            "cabin_target_bad": cabin_target_bad,
            "cabin_constraint_bad": cabin_constraint_bad,
            "nl_fact_preservation_fail": nl_fact_fail,
        },
        "sample_ids": {
            "nl_history_sample_households": [h["household_id"] for h in sampled],
            "route_task_sample": [t["task_id"] for t in rt_samp],
            "cabin_task_sample": [t["task_id"] for t in ct_samp],
        },
        "num_issues": len(issues),
    }
    os.makedirs(f"{ROOT}/reports", exist_ok=True)
    with open(f"{ROOT}/reports/qc_issues.jsonl", "w") as f:
        for sev, check, sid, detail in issues:
            f.write(json.dumps({"severity": sev, "check": check,
                                "sample_id": sid, "detail": detail}) + "\n")

    md = [f"# HabitDrive data QC report\n",
          f"- data dir: `{DATA}`",
          f"- sample seed: {SAMPLE_SEED}",
          f"- total issues: {len(issues)}\n",
          "## Counts (paper target / actual)\n",
          "| metric | paper | actual |", "|---|---|---|"]
    for k, want in PAPER.items():
        md.append(f"| {k} | {want} | {counts[k]} |")
    md += [f"| members | (unspecified) | {counts['members']} |",
           f"| memories | (unspecified) | {counts['memories']} |",
           f"| expanded events | (unspecified) | {counts['expanded_events']} |",
           f"| parking subset | (unspecified) | {counts['route_tasks_with_parking']} |",
           "",
           "## Challenge label counts (multi-label; sum != task total)",
           "| label | count |", "|---|---|"]
    for lab in sorted(LABELS):
        md.append(f"| {lab} | {label_counter.get(lab, 0)} |")
    md += ["", "## Checks", "| check | value |", "|---|---|"]
    for k, v in report["checks"].items():
        md.append(f"| {k} | {v} |")
    md += ["", "## Note on independence",
           "Reference-answer validity re-uses the generator's requirement "
           "resolver (disclosed); all other checks are independent.",
           "", "## Issue list (first 50)", "```"]
    md += [f"{sev}: {check}: {sid}: {detail}"
           for sev, check, sid, detail in issues[:50]]
    md.append("```")
    with open(f"{ROOT}/reports/qc_report.md", "w") as f:
        f.write("\n".join(md))

    print(json.dumps(report, indent=1))
    print(f"\nissues: {len(issues)}")
    for sev, check, sid, detail in issues[:40]:
        print(f"  [{sev}] {check}: {sid}: {detail}")


def gs_fact_preserved(fact, text):
    """Independent light check that key fact tokens survived NL verbalization."""
    blob = text.lower()
    st = fact.get("state", {})
    cat = fact.get("category", "")
    if cat == "location" and st.get("place_name"):
        return st["place_name"].lower() in blob
    if cat in ("climate", "seat", "media", "window", "ambient_light") \
            and st.get("value") is not None:
        return str(st["value"]).lower() in blob
    if cat == "parking_preference" and st.get("lot_name"):
        return st["lot_name"].lower() in blob
    return True


if __name__ == "__main__":
    main()
