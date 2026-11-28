"""Validation and counting for the generated HabitDrive dataset.

Checks paper-fixed totals, per-household base-event counts, unique gold
candidates (recomputed from memory state), challenge-label coverage, and
JSON/chronology integrity.
"""
from __future__ import annotations

import json
from collections import Counter

import constants as C
import generate_structure as gs

ALLOWED_LABELS = {"user_binding", "preference_update",
                  "conditional_application", "temporary_request_interference"}
ALLOWED_EVO_OPS = {"content_update", "condition_update",
                   "unconditional_to_conditional", "conditional_to_unconditional",
                   "cancellation"}


def _build_active(memories):
    out = {}
    for m in memories:
        node = m["trajectory"][-1]
        if node["operation"] != "cancellation":
            out[m["memory_id"]] = (m, node)
    return out


def _route_satisfying(routes, reqs):
    return [r for r in routes
            if all(gs.route_attr_ok(r, a, v) for (_, a, v) in reqs)]


def _parking_satisfying(lots, reqs):
    return [p for p in lots
            if all(gs.parking_attr_ok(p, a, v) for (_, a, v) in reqs)]


def validate(data, verbose=True):
    errors = []
    warn = []

    households = data["households"]
    memories = data["memories"]
    histories = data["histories"]
    route_tasks = data["route_tasks"]
    cabin_tasks = data["cabin_tasks"]

    # ---- Paper-fixed totals ----
    if len(households) != C.NUM_HOUSEHOLDS:
        errors.append(f"households {len(households)} != {C.NUM_HOUSEHOLDS}")
    if len(route_tasks) != 2000:
        errors.append(f"route_tasks {len(route_tasks)} != 2000")
    if len(cabin_tasks) != 1000:
        errors.append(f"cabin_tasks {len(cabin_tasks)} != 1000")

    # ---- Base events ----
    base_total = 0
    for h in histories:
        base = [e for e in h["events"]
                if e["event_type"] in ("establishment", "evolution")]
        base_total += len(base)
        if len(base) != C.BASE_EVENTS_PER_HOUSEHOLD:
            errors.append(f"{h['household_id']}: base events {len(base)} != "
                          f"{C.BASE_EVENTS_PER_HOUSEHOLD}")
    if base_total != 5000:
        errors.append(f"base_events {base_total} != 5000")

    # ---- Membership integrity ----
    member_ids = {m["member_id"] for h in households for m in h["members"]}
    hid_set = {h["household_id"] for h in households}
    for m in memories:
        if m["household_id"] not in hid_set:
            errors.append(f"{m['memory_id']}: bad household")
        if m["owner_id"] not in member_ids:
            errors.append(f"{m['memory_id']}: bad owner {m['owner_id']}")
        if m["category"] not in ("location", "route_preference",
                                 "parking_preference", "passenger_constraint",
                                 *C.CABIN_CATEGORIES.keys()):
            errors.append(f"{m['memory_id']}: bad category {m['category']}")
        for node in m["trajectory"]:
            op = node["operation"]
            if op not in ("establish", *ALLOWED_EVO_OPS):
                errors.append(f"{m['memory_id']}: bad op {op}")

    # ---- Chronology ----
    for h in histories:
        ts = [e["timestamp"] for e in h["events"]]
        if ts != sorted(ts):
            errors.append(f"{h['household_id']}: history not chronological")

    # ---- Route task unique gold (recomputed) ----
    active = _build_active(memories)
    member_of = {m["member_id"]: h["household_id"]
                 for h in households for m in h["members"]}
    parking_tasks = 0
    label_counter = Counter()
    route_label_counter = Counter()
    cabin_label_counter = Counter()

    for t in route_tasks:
        hid = t["household_id"]
        pub = t["public"]
        ev = t["evaluation"]
        requester = pub["request"]["speaker_id"]
        ctx = pub["context"]
        role = ev["gold"]["destination"]["semantic_role"]
        occ = ctx["occupant_ids"]

        reqs, _ = gs.resolve_route_requirements(active, requester, role, ctx, occ)
        if not reqs:
            reqs = [(None, "travel_time", "minimize")]
        routes = pub["stages"]["route_selection"]["candidates"]
        satisfying = _route_satisfying(routes, reqs)
        gold = ev["gold"]["route_id"]
        if len(satisfying) != 1:
            errors.append(f"{t['task_id']}: {len(satisfying)} routes satisfy "
                          f"requirements (want 1)")
        elif satisfying[0]["route_id"] != gold:
            errors.append(f"{t['task_id']}: gold route not the unique "
                          "satisfying route")

        if pub["parking_required"]:
            parking_tasks += 1
            preqs, _ = gs.resolve_parking_requirements(
                active, requester, role, ctx, occ)
            lots = pub["stages"]["parking_selection"]["candidates"]
            psat = _parking_satisfying(lots, preqs)
            goldp = ev["gold"]["parking_id"]
            if len(psat) != 1:
                errors.append(f"{t['task_id']}: {len(psat)} parking satisfy "
                              "(want 1)")
            elif psat[0]["parking_id"] != goldp:
                errors.append(f"{t['task_id']}: gold parking not unique")
        else:
            if ev["gold"]["parking_id"] is not None:
                errors.append(f"{t['task_id']}: parking_id set but not required")
            if pub["stages"]["parking_selection"]["candidates"]:
                errors.append(f"{t['task_id']}: parking candidates but not required")

        for lab in ev["challenge_labels"]:
            if lab not in ALLOWED_LABELS:
                errors.append(f"{t['task_id']}: bad label {lab}")
            label_counter[lab] += 1
            route_label_counter[lab] += 1

    # ---- Cabin task integrity ----
    all_fields = set(gs.all_cabin_fields())
    for t in cabin_tasks:
        ev = t["evaluation"]
        pub = t["public"]
        if not ev["target_fields"]:
            errors.append(f"{t['task_id']}: empty target_fields")
        for f in ev["target_fields"]:
            if f not in all_fields:
                errors.append(f"{t['task_id']}: bad target field {f}")
        protected = set(ev["protected_fields"])
        if protected & set(ev["target_fields"]):
            errors.append(f"{t['task_id']}: protected overlaps target")
        # target fields must differ from initial state (task requires a change)
        init = pub["environment"]["initial_device_state"]
        for f, v in ev["target_fields"].items():
            if init.get(f) == v:
                warn.append(f"{t['task_id']}: target {f} already equals "
                            "initial state")
        for lab in ev["challenge_labels"]:
            if lab not in ALLOWED_LABELS:
                errors.append(f"{t['task_id']}: bad label {lab}")
            label_counter[lab] += 1
            cabin_label_counter[lab] += 1

    # ---- Label coverage ----
    for lab in ALLOWED_LABELS:
        if label_counter[lab] == 0:
            errors.append(f"challenge label '{lab}' has zero instances")

    report = {
        "households": len(households),
        "members": sum(len(h["members"]) for h in households),
        "memories": len(memories),
        "base_events": base_total,
        "route_tasks": len(route_tasks),
        "route_tasks_with_parking": parking_tasks,
        "cabin_tasks": len(cabin_tasks),
        "label_counts": dict(label_counter),
        "route_label_counts": dict(route_label_counter),
        "cabin_label_counts": dict(cabin_label_counter),
        "evolution_op_counts": Counter(
            n["operation"] for m in memories for n in m["trajectory"]
            if n["operation"] != "establish"),
        "category_counts": Counter(m["category"] for m in memories),
        "errors": errors,
        "warnings": warn,
    }

    if verbose:
        print(json.dumps({k: v for k, v in report.items()
                          if k not in ("errors", "warnings")}, indent=1))
        if errors:
            print(f"\nERRORS ({len(errors)}):")
            for e in errors[:50]:
                print("  ", e)
        if warn:
            print(f"\nWARNINGS ({len(warn)}):")
            for w in warn[:20]:
                print("  ", w)
    return report


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Validate from a saved data directory.
        d = sys.argv[1]
        data = {
            "households": json.load(open(f"{d}/households.json"))["households"],
            "memories": json.load(open(f"{d}/memories.json"))["memories"],
            "histories": json.load(open(f"{d}/structured_history.json"))["histories"],
            "route_tasks": json.load(open(f"{d}/route_tasks.json"))["tasks"],
            "cabin_tasks": json.load(open(f"{d}/cabin_tasks.json"))["tasks"],
        }
    else:
        data = gs.generate_structure()
    validate(data, verbose=True)
