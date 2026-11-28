"""Finalize: aggregate per-household verbalized histories, write the final
natural-language history and manifest, and report actual on-disk counts."""
from __future__ import annotations

import json
import os
from collections import Counter

import constants as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
HIST_DIR = os.path.join(DATA, "history")
MODEL_PATH = "/path/to/weights/Llama-3.1-8B-Instruct"

ROUTE_ATTR_KEYWORDS = {
    "highway": ["highway"], "toll_road": ["toll"],
    "congestion": ["congestion"], "traffic_lights": ["traffic", "lights"],
    "lane_changes": ["lane"], "smoothness": ["smooth"],
    "scenery": ["scenic", "scenery"],
    "travel_time": ["fast", "time", "quick", "shortest"],
    "distance": ["distance", "short"],
}
CONSTRAINT_KEYWORDS = {
    "motion_sickness": ["sick", "smooth", "winding"],
    "limited_mobility": ["mobility", "walk"],
    "need_regular_breaks": ["break", "rest"],
}
PARKING_TYPE_KEYWORDS = {
    "nearest": ["nearest", "closest"], "low_cost": ["low-cost", "cheap", "low cost"],
    "indoor": ["indoor"], "easy_to_park": ["easy"],
    "preferred_parking_lot": ["park"],
}


def _blob(u, a):
    return (u + " " + a).lower()


def fact_preserved(fact, u, a):
    blob = _blob(u, a)
    cat = fact.get("category", "")
    st = fact.get("state", {})
    if cat == "location" and st.get("place_name"):
        return st["place_name"].lower() in blob
    if cat == "route_preference":
        kws = ROUTE_ATTR_KEYWORDS.get(st.get("attribute", ""), [])
        return any(k in blob for k in kws)
    if cat == "parking_preference":
        pt = st.get("preference_type", "")
        if st.get("lot_name"):
            return st["lot_name"].lower() in blob
        kws = PARKING_TYPE_KEYWORDS.get(pt, [])
        return any(k in blob for k in kws)
    if cat == "passenger_constraint":
        kws = CONSTRAINT_KEYWORDS.get(st.get("constraint_type", ""), [])
        return any(k in blob for k in kws)
    if cat in ("climate", "seat", "media", "window", "ambient_light"):
        return str(st.get("value", "")).lower() in blob
    if fact.get("operation") == "temporary":
        instr = fact.get("instruction", {})
        attr = instr.get("attribute", "")
        kws = ROUTE_ATTR_KEYWORDS.get(attr, []) or [attr.replace("_", " ")]
        return any(k in blob for k in kws)
    return True


def load(p):
    return json.load(open(p, encoding="utf-8"))


def finalize():
    households = load(f"{DATA}/households.json")["households"]
    structured = load(f"{DATA}/structured_history.json")["histories"]
    route_tasks = load(f"{DATA}/route_tasks.json")["tasks"]
    cabin_tasks = load(f"{DATA}/cabin_tasks.json")["tasks"]

    # Aggregate verbalized per-household histories.
    struct_map = {h["household_id"]: {e["event_id"]: e for e in h["events"]}
                  for h in structured}
    histories = []
    missing = []
    reverted = 0
    for h in households:
        hid = h["household_id"]
        p = f"{HIST_DIR}/{hid}.json"
        if not os.path.exists(p):
            missing.append(hid)
            continue
        vh = load(p)
        smap = struct_map.get(hid, {})
        # Build flat role-based message list for downstream ingestion.
        messages = []
        for e in vh["events"]:
            se = smap.get(e["event_id"], e)
            u = e.get("user_utterance", "")
            a = e.get("agent_response", "")
            if not fact_preserved(se.get("fact", {}), u, a):
                u = se.get("user_utterance", u)
                a = se.get("agent_response", a)
                reverted += 1
            messages.append({
                "role": "user",
                "speaker_id": e["speaker_id"],
                "speaker_name": e.get("speaker_name", ""),
                "text": u,
                "timestamp": e["timestamp"],
                "event_id": e["event_id"],
                "event_type": e["event_type"],
            })
            messages.append({
                "role": "assistant",
                "speaker_id": None,
                "speaker_name": None,
                "text": a,
                "timestamp": e["timestamp"],
                "event_id": e["event_id"],
                "event_type": e["event_type"],
            })
        histories.append({"household_id": hid, "namespace_id": hid,
                          "history": messages})

    num_events = sum(len(h["history"]) // 2 for h in histories)
    meta = {
        "benchmark": "HabitDrive",
        "schema_version": "2.0",
        "seed": C.MASTER_SEED,
        "language": "English",
        "num_households": len(histories),
        "num_turns": sum(len(h["history"]) for h in histories),
        "num_interactions": num_events,
        "model_path": MODEL_PATH,
        "inference_backend": "transformers_direct",
        "verbalized_with_local_model": True,
    }
    with open(f"{DATA}/natural_language_history.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "histories": histories}, f,
                  ensure_ascii=False, indent=1)

    # Counts from disk.
    base_events = sum(1 for h in structured for e in h["events"]
                      if e["event_type"] in ("establishment", "evolution"))
    expanded = sum(1 for h in structured for e in h["events"]
                   if e["event_type"] not in ("establishment", "evolution"))
    parking = sum(1 for t in route_tasks if t["public"]["parking_required"])
    route_labels = Counter(lab for t in route_tasks
                           for lab in t["evaluation"]["challenge_labels"])
    cabin_labels = Counter(lab for t in cabin_tasks
                           for lab in t["evaluation"]["challenge_labels"])
    total_labels = route_labels + cabin_labels

    manifest = {
        "benchmark": "HabitDrive",
        "paper_targets": {
            "households": 100, "base_history_events": 5000,
            "route_tasks": 2000, "cabin_tasks": 1000,
        },
        "actual_counts": {
            "households": len(households),
            "members": sum(len(h["members"]) for h in households),
            "memories": len(load(f"{DATA}/memories.json")["memories"]),
            "base_history_events": base_events,
            "expanded_history_events": expanded,
            "total_history_interactions": num_events,
            "route_tasks": len(route_tasks),
            "route_tasks_with_parking": parking,
            "cabin_tasks": len(cabin_tasks),
        },
        "challenge_label_counts": dict(total_labels),
        "route_challenge_label_counts": dict(route_labels),
        "cabin_challenge_label_counts": dict(cabin_labels),
        "verbalized_households": len(histories),
        "missing_verbalized_households": missing,
        "reverted_to_template_events": reverted,
        "seed": C.MASTER_SEED,
        "model": {"path": MODEL_PATH, "name": "Llama-3.1-8B-Instruct",
                  "dtype": "bfloat16"},
        "decisions": "See GENERATION_NOTES.md",
    }
    with open(f"{DATA}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    print(json.dumps(manifest, indent=1))
    if missing:
        print("\nMISSING households:", missing)
    return manifest, missing


if __name__ == "__main__":
    finalize()
