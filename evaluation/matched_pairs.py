"""Construct factor-isolating matched pairs (counterfactual variants).

For the conditional-application factor: take route tasks where a
weather-conditional preference is ACTIVE (rainy/snowy/foggy), build a
counterfactual with weather=clear (condition inactive), and recompute the
requirements + gold route for the counterfactual. Running the decision model
on both base and counterfactual then shows whether the agent adapts to the
condition.

CPU-only; the pairs + recomputed gold are saved for the runner.
"""
from __future__ import annotations

import json
import os
import sys

GEN = "/path/to/habitdrive_full/code"
DATA = "/path/to/habitdrive_full/data"
ROOT = "/path/to/habitdrive_eval"

sys.path.insert(0, GEN)
import generate_structure as gs  # noqa: E402


def _weather_conditional(m, node):
    return (m["category"] == "route_preference"
            and node["condition"]
            and "weather" in node["condition"])


def build(n_pairs=15, seed=20251127):
    import random
    rng = random.Random(seed)
    memories = json.load(open(f"{DATA}/memories.json", encoding="utf-8"))["memories"]
    route = json.load(open(f"{DATA}/route_tasks.json", encoding="utf-8"))["tasks"]
    active = gs.final_active(memories)

    pairs = []
    for t in route:
        pub, ev = t["public"], t["evaluation"]
        ctx = pub["context"]
        if ctx.get("weather") not in ("rainy", "snowy", "foggy"):
            continue
        requester = pub["request"]["speaker_id"]
        role = ev["gold"]["destination"]["semantic_role"]
        occ = ctx["occupant_ids"]
        reqs, _ = gs.resolve_route_requirements(active, requester, role, ctx, occ)
        # a weather-conditional pref must be active in the base context
        if not any(_weather_conditional(m, n) for (m, n) in active.values()
                   if m["owner_id"] == requester
                   and gs.condition_active(n["condition"], role, ctx)):
            continue
        # counterfactual: weather -> clear (condition inactive)
        ctx2 = dict(ctx)
        ctx2["weather"] = "clear"
        reqs2, _ = gs.resolve_route_requirements(active, requester, role, ctx2, occ)
        if not reqs2:
            reqs2 = [(None, "travel_time", "minimize")]
        routes2, gold2 = gs.generate_route_candidates(rng, reqs2)
        # assign route ids and find the counterfactual gold id
        gold2_id = None
        for i, r in enumerate(routes2, 1):
            r["route_id"] = f"{t['task_id']}_cf_route_{i}"
            if r is gold2:
                gold2_id = r["route_id"]
        pairs.append({
            "task_id": t["task_id"],
            "household_id": t["household_id"],
            "base_context": ctx,
            "counterfactual_context": ctx2,
            "base_gold_route": ev["gold"]["route_id"],
            "counterfactual_gold_route": gold2_id,
            "counterfactual_candidates": routes2,
            "request": pub["request"],
        })
        if len(pairs) >= n_pairs:
            break

    out = f"{ROOT}/data/smoke/matched_pairs.json"
    os.makedirs(f"{ROOT}/data/smoke", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_pairs": len(pairs), "seed": seed, "pairs": pairs}, f,
                  ensure_ascii=False, indent=1)
    print(f"built {len(pairs)} matched pairs -> {out}")
    for p in pairs[:5]:
        print("  ", p["task_id"], "base_weather=", p["base_context"]["weather"],
              "-> counterfactual_weather=clear")


if __name__ == "__main__":
    build()
