"""Orchestrator: generate structured data, save to disk, and validate.

Usage:
  python run.py structure   # generate + save + validate structured data
  python run.py verbalize   # (delegated to verbalize.py, run on GPU)
  python run.py validate    # validate from saved data directory
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter

import constants as C
import generate_structure as gs
import validate as val

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data")
MODEL_PATH = "/path/to/weights/Llama-3.1-8B-Instruct"


def _metadata(extra):
    return {
        "benchmark": "HabitDrive",
        "schema_version": "2.0",
        "seed": C.MASTER_SEED,
        "language": "English",
        **extra,
    }


def save(data):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "history"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "progress"), exist_ok=True)

    households = data["households"]
    memories = data["memories"]
    histories = data["histories"]
    route_tasks = data["route_tasks"]
    cabin_tasks = data["cabin_tasks"]

    base_events = sum(1 for h in histories for e in h["events"]
                      if e["event_type"] in ("establishment", "evolution"))
    expanded_events = sum(1 for h in histories for e in h["events"]
                          if e["event_type"] not in ("establishment", "evolution"))

    with open(f"{DATA_DIR}/households.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": _metadata({"num_households": len(households),
                                          "num_members": sum(len(h["members"]) for h in households)}),
                   "households": households}, f, ensure_ascii=False, indent=1)

    with open(f"{DATA_DIR}/memories.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": _metadata({"num_memories": len(memories)}),
                   "memories": memories}, f, ensure_ascii=False, indent=1)

    with open(f"{DATA_DIR}/structured_history.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": _metadata({"num_households": len(histories),
                                          "num_events": sum(len(h["events"]) for h in histories),
                                          "num_base_events": base_events,
                                          "num_expanded_events": expanded_events}),
                   "histories": histories}, f, ensure_ascii=False, indent=1)

    with open(f"{DATA_DIR}/route_tasks.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": _metadata({"num_tasks": len(route_tasks)}),
                   "tasks": route_tasks}, f, ensure_ascii=False, indent=1)

    with open(f"{DATA_DIR}/cabin_tasks.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": _metadata({"num_tasks": len(cabin_tasks)}),
                   "tasks": cabin_tasks}, f, ensure_ascii=False, indent=1)

    # Public-only views (no gold / evidence / labels) for downstream ingestion.
    route_public = [t["public"] | {"task_id": t["task_id"],
                                   "household_id": t["household_id"]}
                    for t in route_tasks]
    cabin_public = [t["public"] | {"task_id": t["task_id"],
                                   "household_id": t["household_id"]}
                    for t in cabin_tasks]
    with open(f"{DATA_DIR}/route_tasks_public.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": _metadata({"num_tasks": len(route_public)}),
                   "tasks": route_public}, f, ensure_ascii=False, indent=1)
    with open(f"{DATA_DIR}/cabin_tasks_public.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": _metadata({"num_tasks": len(cabin_public)}),
                   "tasks": cabin_public}, f, ensure_ascii=False, indent=1)

    print(f"Saved structured data under {DATA_DIR}")
    return base_events, expanded_events


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "structure"
    if cmd == "structure":
        t0 = time.time()
        data = gs.generate_structure()
        base, expanded = save(data)
        report = val.validate(data, verbose=True)
        print(f"\nGenerated in {time.time()-t0:.1f}s. base_events={base} "
              f"expanded_events={expanded} errors={len(report['errors'])}")
        if report["errors"]:
            sys.exit(1)
    elif cmd == "validate":
        data = {
            "households": json.load(open(f"{DATA_DIR}/households.json"))["households"],
            "memories": json.load(open(f"{DATA_DIR}/memories.json"))["memories"],
            "histories": json.load(open(f"{DATA_DIR}/structured_history.json"))["histories"],
            "route_tasks": json.load(open(f"{DATA_DIR}/route_tasks.json"))["tasks"],
            "cabin_tasks": json.load(open(f"{DATA_DIR}/cabin_tasks.json"))["tasks"],
        }
        report = val.validate(data, verbose=True)
        sys.exit(1 if report["errors"] else 0)
    else:
        print("unknown command", cmd)
        sys.exit(2)


if __name__ == "__main__":
    main()
