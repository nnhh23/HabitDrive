"""Run the two token-matched supplementary baselines (Recent History and
Full Context @8k) through the same smoke pipeline, on GPU."""
from __future__ import annotations

import argparse
import json
import os
import time

import scoring
import smoke_manifest
from smoke import load_data, run_method
from methods import RecentHistory, FullContextTokenMatched
from decision import LocalLLM

ROOT = "/path/to/habitdrive_eval"
OUT = f"{ROOT}/data/outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    households, hist_by_hid, task_by_id = load_data()
    manifest = smoke_manifest.build()
    llm = LocalLLM(gpu=args.gpu, max_input_tokens=9000)
    methods = [RecentHistory(), FullContextTokenMatched()]

    summary = []
    for method in methods:
        t0 = time.time()
        print(f"== running {method.name} ==", flush=True)
        per_task, metrics = run_method(method, manifest, hist_by_hid,
                                       task_by_id, llm)
        out = {"method": method.name, "status": "e2e_run", "error": None,
               "model": "Llama-3.1-8B-Instruct (substitute)",
               "metrics": metrics, "tasks": per_task}
        with open(f"{OUT}/{method.name}.json", "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        n_exc = sum(1 for t in per_task if t["status"] != "executed")
        n_parse = sum(1 for t in per_task if t.get("parse_failure"))
        summary.append({"method": method.name, "status": "e2e_run",
                        "error": None, "n_tasks": len(per_task),
                        "n_exceptions": n_exc, "n_parse_fail": n_parse,
                        "metrics": metrics, "elapsed_s": round(time.time()-t0, 1)})
        print(f"   {method.name}: exc={n_exc} parse={n_parse} "
              f"({time.time()-t0:.0f}s)", flush=True)

    with open(f"{OUT}/supplementary_summary.json", "w") as f:
        json.dump({"manifest": manifest,
                   "model": "Llama-3.1-8B-Instruct (substitute)",
                   "summary": summary}, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
