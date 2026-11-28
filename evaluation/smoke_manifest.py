"""Build the smoke manifest: a fixed, seeded subset of households and tasks
covering route (with/without parking), cabin, all four challenge labels, and
normal (unlabeled) tasks. All methods use this same manifest."""
from __future__ import annotations

import json
import random

DATA = "/path/to/habitdrive_full/data"
ROOT = "/path/to/habitdrive_eval"
SEED = 20251126
LABELS = {"user_binding", "preference_update", "conditional_application",
          "temporary_request_interference"}


def build(num_households=4, tasks_per_household=9):
    rng = random.Random(SEED)
    households = json.load(open(f"{DATA}/households.json"))["households"]
    rt = json.load(open(f"{DATA}/route_tasks.json"))["tasks"]
    ct = json.load(open(f"{DATA}/cabin_tasks.json"))["tasks"]

    by_hid = {}
    for t in rt:
        by_hid.setdefault(t["household_id"], {"route": [], "cabin": []})["route"].append(t)
    for t in ct:
        by_hid.setdefault(t["household_id"], {"route": [], "cabin": []})["cabin"].append(t)

    # pick households deterministically (prefer variety in composition)
    rng.shuffle(households)
    chosen = households[:num_households]

    manifest = {"seed": SEED, "households": [], "tasks": []}
    for h in chosen:
        hid = h["household_id"]
        manifest["households"].append(hid)
        routes = by_hid[hid]["route"]
        cabins = by_hid[hid]["cabin"]
        # split routes into with/without parking
        rp = [t for t in routes if t["public"]["parking_required"]]
        rn = [t for t in routes if not t["public"]["parking_required"]]

        picked = []
        def take(pool, n):
            rng.shuffle(pool)
            picked.extend(pool[:n])

        take(rp, 3)   # parking routes
        take(rn, 2)   # no-parking routes
        take(cabins, 4)  # cabin
        # ensure challenge-label coverage and a couple normal tasks
        for lab in LABELS:
            cand = [t for t in routes + cabins
                    if lab in t["evaluation"]["challenge_labels"]
                    and t["task_id"] not in {p["task_id"] for p in picked}]
            if cand:
                rng.shuffle(cand)
                picked.append(cand[0])
        # normal (unlabeled) tasks
        normal = [t for t in routes + cabins
                  if not t["evaluation"]["challenge_labels"]
                  and t["task_id"] not in {p["task_id"] for p in picked}]
        if normal:
            rng.shuffle(normal)
            picked.extend(normal[:1])

        # trim to tasks_per_household
        picked = picked[:tasks_per_household]
        for t in picked:
            manifest["tasks"].append({
                "task_id": t["task_id"],
                "household_id": t["household_id"],
                "type": "route" if t in routes else "cabin",
                "parking_required": t["public"]["parking_required"]
                if "parking_required" in t["public"] else None,
                "challenge_labels": t["evaluation"]["challenge_labels"],
            })

    return manifest


if __name__ == "__main__":
    m = build()
    import collections
    types = collections.Counter(t["type"] for t in m["tasks"])
    labels = collections.Counter(l for t in m["tasks"] for l in t["challenge_labels"])
    parking = sum(1 for t in m["tasks"] if t["parking_required"])
    normal = sum(1 for t in m["tasks"] if not t["challenge_labels"])
    print(json.dumps(m, indent=1))
    print("\nsummary: types=", dict(types), "parking=", parking,
          "normal=", normal, "labels=", dict(labels),
          "n_tasks=", len(m["tasks"]), "n_households=", len(m["households"]))
