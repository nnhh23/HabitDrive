"""Matched-pair experiment: run the decision model on BOTH sides of each
counterfactual pair (base context vs condition-inactive context) using the SAME
oracle memory (which still contains the conditional fact and its condition).
Measures whether the model adapts to the conditional factor.
"""
from __future__ import annotations

import json

from decision import LocalLLM, extract_json
from methods import OracleMemory

ROOT = "/path/to/habitdrive_eval"
OUT = f"{ROOT}/data/outputs"
DATA = "/path/to/habitdrive_full/data"


def route_prompt(context, candidates, memory_text, request):
    req = request
    cand_txt = "\n".join(
        f"{r['route_id']}: highway={r['uses_highway']}, "
        f"toll={r['uses_toll_road']}, congestion={r['congestion_level']}, "
        f"traffic_lights={r['traffic_lights']}, lane_changes={r['lane_changes']}, "
        f"smoothness={r['smoothness']}, scenery={r['scenery']}, "
        f"time={r['travel_time_minutes']}min, distance={r['distance_km']}km, "
        f"rest_stop={r['rest_stop_available']}" for r in candidates)
    return (
        f"Trip request: \"{req['text']}\" (speaker {req['speaker_name']}).\n"
        f"Context: weather={context.get('weather')}, time={context.get('time_period')}, "
        f"trip={context.get('trip_context')}.\n"
        f"\nRemembered information:\n{memory_text}\n"
        f"\nCandidate routes:\n{cand_txt}\n"
        f"\nSelect the route that satisfies the user's preferences. "
        f"Output ONLY JSON: {{\"route_id\": \"<id>\"}}."
    )


def main(gpu=0):
    pairs = json.load(open(f"{ROOT}/data/smoke/matched_pairs.json",
                           encoding="utf-8"))["pairs"]
    route_tasks = {t["task_id"]: t for t in json.load(
        open(f"{DATA}/route_tasks.json", encoding="utf-8"))["tasks"]}

    oracle = OracleMemory()
    oracle.ingest("x", [], None)

    llm = LocalLLM(gpu=gpu, max_input_tokens=3500)
    # build all prompts (base + counterfactual)
    prompts = []
    meta = []   # (task_id, side)
    for p in pairs:
        bt = route_tasks[p["task_id"]]
        mem = oracle.provide(bt, "route_selection")
        base_cand = bt["public"]["stages"]["route_selection"]["candidates"]
        prompts.append(route_prompt(p["base_context"], base_cand, mem,
                                    p["request"]))
        meta.append((p["task_id"], "base"))
        prompts.append(route_prompt(p["counterfactual_context"],
                                    p["counterfactual_candidates"], mem,
                                    p["request"]))
        meta.append((p["task_id"], "counterfactual"))

    raws = []
    for i in range(0, len(prompts), 8):
        raws.extend(llm.generate(prompts[i:i + 8]))

    # aggregate per pair
    by_pair = {}
    for (tid, side), raw in zip(meta, raws):
        obj = extract_json(raw) or {}
        by_pair.setdefault(tid, {})[side] = obj.get("route_id", "")

    correct_base = correct_cf = adapt = 0
    rows = []
    for p in pairs:
        base_chosen = by_pair[p["task_id"]]["base"]
        cf_chosen = by_pair[p["task_id"]]["counterfactual"]
        base_ok = (base_chosen == p["base_gold_route"])
        cf_ok = (cf_chosen == p["counterfactual_gold_route"])
        correct_base += 1 if base_ok else 0
        correct_cf += 1 if cf_ok else 0
        adapt += 1 if (base_ok and cf_ok) else 0
        rows.append({"task_id": p["task_id"], "base_chosen": base_chosen,
                     "base_gold": p["base_gold_route"], "base_ok": base_ok,
                     "cf_chosen": cf_chosen,
                     "cf_gold": p["counterfactual_gold_route"], "cf_ok": cf_ok})

    n = len(pairs)
    result = {"n_pairs": n, "base_correct": correct_base,
              "counterfactual_correct": correct_cf,
              "adaptation": adapt, "adaptation_rate": adapt / n if n else None,
              "rows": rows}
    with open(f"{OUT}/matched_results.json", "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"pairs={n} base_correct={correct_base}/{n} "
          f"counterfactual_correct={correct_cf}/{n} "
          f"adaptation={adapt}/{n} ({adapt/n*100:.0f}%)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    main(ap.parse_args().gpu)
