"""Smoke run: run every method through history-ingest -> memory -> decision
-> parse -> score, saving per-task detail and per-method metrics.

Usage:
  python smoke.py --dummy              # offline pipeline test (no GPU/model)
  python smoke.py --gpu 0              # real local Llama-3.1-8B decision model
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import scoring
import smoke_manifest
from methods import build_basic_methods, build_representative_methods

DATA = "/path/to/habitdrive_full/data"
ROOT = "/path/to/habitdrive_eval"
OUT = f"{ROOT}/data/outputs"


class DummyLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompts):
        self.calls += len(prompts)
        return ['{"destination": "UNKNOWN"}' for _ in prompts]


def load_data():
    households = json.load(open(f"{DATA}/households.json"))["households"]
    nl = json.load(open(f"{DATA}/natural_language_history.json"))["histories"]
    route = json.load(open(f"{DATA}/route_tasks.json"))["tasks"]
    cabin = json.load(open(f"{DATA}/cabin_tasks.json"))["tasks"]
    hist_by_hid = {h["household_id"]: h["history"] for h in nl}
    task_by_id = {}
    for t in route:
        t["_type"] = "route"
        task_by_id[t["task_id"]] = t
    for t in cabin:
        t["_type"] = "cabin"
        task_by_id[t["task_id"]] = t
    return households, hist_by_hid, task_by_id


def run_route_task(method, task, llm, record):
    from decision import (route_destination_prompt, route_selection_prompt,
                          parking_prompt, extract_json)
    pub = task["public"]
    # stage 1: destination
    mem = method.provide(task, "destination")
    p = route_destination_prompt(task, mem)
    raw = llm.generate([p])[0]
    obj = extract_json(raw) or {}
    dest = obj.get("destination", "")
    record["stages"]["destination"] = {"memory": mem, "prompt": p,
                                       "raw": raw, "parsed": dest}
    dest_ok = (dest.strip().lower() ==
               task["evaluation"]["gold"]["destination"]["name"].lower())
    if not dest_ok:
        record["stop_reason"] = "destination_wrong"
        return

    # stage 2: route
    mem = method.provide(task, "route_selection")
    p = route_selection_prompt(task, mem)
    raw = llm.generate([p])[0]
    obj = extract_json(raw) or {}
    route_id = obj.get("route_id", "")
    record["stages"]["route"] = {"memory": mem, "prompt": p, "raw": raw,
                                 "parsed": route_id}
    if route_id != task["evaluation"]["gold"]["route_id"]:
        record["stop_reason"] = "route_wrong"
        return

    # stage 3: parking (only if required)
    if pub["parking_required"]:
        mem = method.provide(task, "parking_selection")
        p = parking_prompt(task, mem)
        raw = llm.generate([p])[0]
        obj = extract_json(raw) or {}
        parking_id = obj.get("parking_id", "")
        record["stages"]["parking"] = {"memory": mem, "prompt": p, "raw": raw,
                                       "parsed": parking_id}
        if parking_id != task["evaluation"]["gold"]["parking_id"]:
            record["stop_reason"] = "parking_wrong"
            return
    record["stop_reason"] = "success"


def run_cabin_task(method, task, llm, record):
    from decision import cabin_prompt, extract_json
    mem = method.provide(task, "cabin")
    p = cabin_prompt(task, mem)
    raw = llm.generate([p])[0]
    obj = extract_json(raw) or {}
    record["stages"]["cabin"] = {"memory": mem, "prompt": p, "raw": raw,
                                 "parsed": obj}
    record["stop_reason"] = "done"


def run_method(method, manifest, hist_by_hid, task_by_id, llm):
    tasks = manifest["tasks"]
    # ingest each household once
    for hid in manifest["households"]:
        method.reset(hid)
        method.ingest(hid, hist_by_hid.get(hid, []), llm)

    # A blocked method (e.g. representative system whose backend is
    # unavailable) must not be scored with empty memory.
    status = getattr(method, "status", "e2e_run")
    if status == "blocked":
        per_task = [{
            "task_id": mt["task_id"], "household_id": mt["household_id"],
            "type": mt["type"], "method": method.name,
            "status": "blocked", "error": getattr(method, "error", None),
            "score": {}, "decision": {},
        } for mt in tasks]
        return per_task, None

    per_task = []
    for mt in tasks:
        task = task_by_id[mt["task_id"]]
        record = {
            "task_id": mt["task_id"], "household_id": mt["household_id"],
            "type": mt["type"], "method": method.name,
            "stages": {}, "decision": {}, "score": {}, "parse_failure": False,
            "status": "executed",
        }
        try:
            if mt["type"] == "route":
                run_route_task(method, task, llm, record)
                dec = {"destination": record["stages"].get("destination", {}).get("parsed", ""),
                       "route_id": record["stages"].get("route", {}).get("parsed", ""),
                       "parking_id": record["stages"].get("parking", {}).get("parsed", "")}
                record["decision"] = dec
                record["score"] = scoring.score_route_task(dec, task)
            else:
                run_cabin_task(method, task, llm, record)
                obj = record["stages"].get("cabin", {}).get("parsed", {})
                if not isinstance(obj, dict):
                    obj = {}
                    record["parse_failure"] = True
                record["decision"] = obj
                record["score"] = scoring.score_cabin_task(obj, task)
        except Exception as e:
            record["status"] = "exception"
            record["error"] = f"{type(e).__name__}: {e}"
        per_task.append(record)

    # metrics
    if method.name != "no_memory" or True:
        r_pts = [t["score"] for t in per_task if t["type"] == "route"
                 and "destination_correct" in t["score"]]
        c_pts = [t["score"] for t in per_task if t["type"] == "cabin"
                 and "field_correct" in t["score"]]
        metrics = {
            "route": scoring.route_metrics(r_pts) if r_pts else None,
            "cabin": scoring.cabin_metrics(c_pts) if c_pts else None,
        }
    return per_task, metrics


def _fmt(history_messages):
    from decision import fmt_history
    return fmt_history(history_messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--basic-only", action="store_true")
    args = ap.parse_args()

    households, hist_by_hid, task_by_id = load_data()
    manifest = smoke_manifest.build()
    os.makedirs(OUT, exist_ok=True)
    with open(f"{ROOT}/data/smoke/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)

    if args.dummy:
        llm = DummyLLM()
        model_desc = "dummy"
    else:
        from decision import LocalLLM
        llm = LocalLLM(gpu=args.gpu)
        model_desc = "Llama-3.1-8B-Instruct (substitute)"

    methods = build_basic_methods()
    if not args.basic_only:
        methods += build_representative_methods()

    summary = []
    for method in methods:
        t0 = time.time()
        print(f"== running {method.name} ==", flush=True)
        per_task, metrics = run_method(method, manifest, hist_by_hid,
                                       task_by_id, llm)
        status = getattr(method, "status", "e2e_run")
        out = {
            "method": method.name, "status": status,
            "error": getattr(method, "error", None),
            "model": model_desc, "metrics": metrics, "tasks": per_task,
        }
        with open(f"{OUT}/{method.name}.json", "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        n_exc = sum(1 for t in per_task if t["status"] != "executed")
        n_parse = sum(1 for t in per_task if t.get("parse_failure"))
        summary.append({"method": method.name, "status": status,
                        "error": getattr(method, "error", None),
                        "n_tasks": len(per_task), "n_exceptions": n_exc,
                        "n_parse_fail": n_parse,
                        "metrics": metrics,
                        "elapsed_s": round(time.time() - t0, 1)})
        print(f"   {method.name}: status={status} exc={n_exc} "
              f"parse_fail={n_parse} ({time.time()-t0:.0f}s)", flush=True)

    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({"manifest": manifest, "model": model_desc,
                   "summary": summary}, f, indent=1, ensure_ascii=False)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
