"""HabitDrive scoring: E2E, DA, PSA, TFA (per paper Section 4.1.2).

Route task success = destination correct AND route correct AND (parking
correct if required).  Cabin task success = all target fields correct AND
protected fields unchanged AND constraints satisfied.
"""
from __future__ import annotations


def score_route_task(decision, task):
    """decision: {destination, route_id, parking_id}. Returns per-task detail."""
    ev = task["evaluation"]
    pub = task["public"]
    gold = ev["gold"]
    parking_required = pub["parking_required"]

    d = decision.get("destination", "").strip()
    dest_correct = (d.lower() == gold["destination"]["name"].lower())
    route_correct = (decision.get("route_id") == gold["route_id"])
    if parking_required:
        parking_correct = (decision.get("parking_id") == gold["parking_id"])
    else:
        parking_correct = True  # P_i = 1 when parking not required
    e2e = dest_correct and route_correct and parking_correct
    return {
        "destination_correct": bool(dest_correct),
        "route_correct": bool(route_correct),
        "parking_correct": bool(parking_correct),
        "parking_required": bool(parking_required),
        "e2e": bool(e2e),
    }


def route_metrics(per_task):
    """per_task: list of score_route_task results. Returns dict of metrics."""
    N = len(per_task)
    D = sum(1 for t in per_task if t["destination_correct"])
    correct_dest_idx = [t for t in per_task if t["destination_correct"]]
    # PSA numerator: tasks with correct destination AND (correct route, and
    # correct parking if required).
    psa_num = sum(1 for t in correct_dest_idx
                  if t["route_correct"] and t["parking_correct"])
    psa_den = len(correct_dest_idx)
    e2e = sum(1 for t in per_task if t["e2e"])
    return {
        "N": N,
        "DA": (D / N) if N else None,
        "DA_num": D,
        "DA_den": N,
        "PSA": (psa_num / psa_den) if psa_den else None,
        "PSA_num": psa_num,
        "PSA_den": psa_den,
        "E2E": (e2e / N) if N else None,
        "E2E_num": e2e,
        "E2E_den": N,
    }


def score_cabin_task(decision, task):
    """decision: {settings: {field: value}}. Applies plan to initial state,
    then checks targets, protected fields, and constraints."""
    ev = task["evaluation"]
    pub = task["public"]
    init = pub["environment"]["initial_device_state"]
    targets = ev["target_fields"]
    protected = set(ev["protected_fields"])

    plan = decision.get("settings", {}) or {}
    final = dict(init)
    for f, v in plan.items():
        if f in final or f in ev["target_fields"] or f in protected:
            final[f] = v

    field_correct = {f: (final.get(f) == v) for f, v in targets.items()}
    protected_violated = [f for f in protected if final.get(f) != init.get(f)]
    constraint_violated = []
    for c in ev["execution_constraints"]:
        if c["type"] == "mutual_exclusion":
            a, b = c["fields"]
            for zone in ("driver", "front_passenger", "rear"):
                va = final.get(f"{zone}.{a}", "off")
                vb = final.get(f"{zone}.{b}", "off")
                if va != "off" and vb != "off":
                    constraint_violated.append(f"{zone}.{a}/{b}")

    n_target = len(targets)
    n_correct = sum(1 for v in field_correct.values() if v)
    tfa = (n_correct / n_target) if n_target else None
    e2e = (n_correct == n_target) and not protected_violated and \
        not constraint_violated
    return {
        "field_correct": field_correct,
        "n_correct": n_correct,
        "n_target": n_target,
        "TFA": tfa,
        "protected_violated": protected_violated,
        "constraint_violated": constraint_violated,
        "e2e": bool(e2e),
    }


def cabin_metrics(per_task):
    N = len(per_task)
    total_fields = sum(t["n_target"] for t in per_task)
    correct_fields = sum(t["n_correct"] for t in per_task)
    e2e = sum(1 for t in per_task if t["e2e"])
    # TFA = mean over tasks of (fraction of target fields correct)
    tfa_mean = (sum(t["TFA"] for t in per_task) / N) if N else None
    return {
        "N": N,
        "TFA": tfa_mean,
        "TFA_num_fields": correct_fields,
        "TFA_den_fields": total_fields,
        "E2E": (e2e / N) if N else None,
        "E2E_num": e2e,
        "E2E_den": N,
    }


def self_test():
    """Cover: fully correct, wrong destination, wrong route after correct
    destination, wrong parking, protected-field change, constraint violation,
    unparseable output."""
    # synthetic route task
    task = {
        "evaluation": {"gold": {"destination": {"name": "River North Office"},
                                "route_id": "r1", "parking_id": "p1"}},
        "public": {"parking_required": True},
    }
    cases = [
        ({"destination": "River North Office", "route_id": "r1", "parking_id": "p1"}, True),
        ({"destination": "Downtown Office", "route_id": "r1", "parking_id": "p1"}, False),
        ({"destination": "River North Office", "route_id": "r2", "parking_id": "p1"}, False),
        ({"destination": "River North Office", "route_id": "r1", "parking_id": "p2"}, False),
        ({"destination": "river north office", "route_id": "r1", "parking_id": "p1"}, True),
        ({}, False),
    ]
    for dec, expect in cases:
        r = score_route_task(dec, task)
        assert r["e2e"] == expect, (dec, r)

    # synthetic cabin task
    ct = {
        "evaluation": {
            "target_fields": {"driver.temperature_c": 22, "driver.seat_heating": "low"},
            "protected_fields": ["passenger.temperature_c", "driver.seat_ventilation"],
            "execution_constraints": [{"type": "mutual_exclusion",
                                       "fields": ["seat_heating", "seat_ventilation"]}],
        },
        "public": {"environment": {
            "initial_device_state": {
                "driver.temperature_c": 20, "driver.seat_heating": "off",
                "driver.seat_ventilation": "off",
                "passenger.temperature_c": 24},
        }},
    }
    # fully correct
    r = score_cabin_task({"settings": {"driver.temperature_c": 22,
                                       "driver.seat_heating": "low"}}, ct)
    assert r["e2e"] and r["TFA"] == 1.0, r
    # correct targets but violates protected field
    r = score_cabin_task({"settings": {"driver.temperature_c": 22,
                                       "driver.seat_heating": "low",
                                       "passenger.temperature_c": 20}}, ct)
    assert not r["e2e"] and "passenger.temperature_c" in r["protected_violated"], r
    # correct targets but constraint violation (ventilation also on)
    r = score_cabin_task({"settings": {"driver.temperature_c": 22,
                                       "driver.seat_heating": "low",
                                       "driver.seat_ventilation": "low"}}, ct)
    assert not r["e2e"] and r["constraint_violated"], r
    # unparseable (empty plan) -> wrong target fields
    r = score_cabin_task({}, ct)
    assert not r["e2e"] and r["TFA"] == 0.0, r
    # wrong destination then wrong route later: DA and PSA denominators
    pts = [
        {"destination_correct": False, "route_correct": False, "parking_correct": False, "parking_required": True, "e2e": False},
        {"destination_correct": True, "route_correct": False, "parking_correct": False, "parking_required": True, "e2e": False},
        {"destination_correct": True, "route_correct": True, "parking_correct": True, "parking_required": True, "e2e": True},
    ]
    m = route_metrics(pts)
    assert m["DA_den"] == 3 and m["DA_num"] == 2 and m["PSA_den"] == 2 and m["PSA_num"] == 1, m
    print("scoring self-test OK")


if __name__ == "__main__":
    self_test()
