"""Generate machine-readable run manifest + human-readable config report."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime

ROOT = "/path/to/habitdrive_eval"
DATA = "/path/to/habitdrive_full/data"
OUT = f"{ROOT}/data/outputs"


def env_facts():
    facts = {}
    facts["os"] = subprocess.run(
        ["grep", "-E", "PRETTY_NAME", "/etc/os-release"],
        capture_output=True, text=True).stdout.strip()
    facts["python"] = sys.version.split()[0]
    import torch
    facts["torch"] = torch.__version__
    facts["cuda"] = torch.version.cuda
    facts["cuda_available_on_login"] = torch.cuda.is_available()
    return facts


def method_status():
    # from smoke summary + representative status file if present
    summ = json.load(open(f"{OUT}/summary.json", encoding="utf-8"))
    st = {}
    for m in summ["summary"]:
        st[m["method"]] = m
    return summ, st


def build_manifest():
    summ, st = method_status()
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task": "smoke_end_to_end (no full evaluation)",
        "data": {
            "input_dir": DATA,
            "version": "v2 (fixed cabin mutual-exclusion; see QC report)",
            "qc_report": f"{ROOT}/reports/qc_report.md",
            "smoke_manifest": f"{ROOT}/data/smoke/manifest.json",
            "smoke_manifest_seed": 20251126,
        },
        "environment": env_facts(),
        "models": {
            "decision_model": {
                "role": "decision",
                "paper": "DeepSeek V4 Flash / Gemini 3.5 Flash-Lite",
                "used": "Llama-3.1-8B-Instruct (local, transformers)",
                "is_paper_config": False,
                "note": "paper models unreachable (gateway down); substitute backend",
                "path": "/path/to/weights/Llama-3.1-8B-Instruct",
                "dtype": "bfloat16",
                "decoding": "temperature 0, max_new_tokens 256",
            },
            "memory_construction_model": {
                "paper": "Qwen3-32B non-thinking",
                "used": "Llama-3.1-8B-Instruct (local) for RS only; "
                        "basic methods are non-parametric",
                "is_paper_config": False,
                "note": "substitute backend",
            },
            "embedding_model": {
                "paper": "BAAI/bge-m3 dense embeddings",
                "used": "keyword matching (scikit-learn/TF-IDF unavailable; "
                        "no local embedding model; gateway down)",
                "is_paper_config": False,
                "note": "substitute backend",
            },
        },
        "methods": {m: {"status": v.get("status"), "error": v.get("error"),
                        "n_tasks": v.get("n_tasks"),
                        "n_exceptions": v.get("n_exceptions"),
                        "n_parse_fail": v.get("n_parse_fail")}
                    for m, v in st.items()},
        "scoring": {
            "entry": f"{ROOT}/code/scoring.py",
            "metrics": "E2E, DA, PSA (conditional denominator), TFA",
            "denominator_rules": "PSA denominator = tasks with correct destination; "
                                 "E2E route = destination AND route AND (parking if "
                                 "required); TFA = mean over tasks of correct target "
                                 "field fraction; cabin E2E = all targets correct AND "
                                 "protected unchanged AND constraints satisfied",
        },
        "attribution": {
            "entry": f"{ROOT}/code/attribution.py",
            "rule": "first failed stage (route) / incorrect field or violation "
                    "(cabin); needed vs seen evidence; acquisition vs decision vs "
                    "undeterminable",
        },
        "review": {
            "table": f"{ROOT}/reports/review_table.csv",
            "agreement_script": f"{ROOT}/code/agreement.py",
            "human_annotation": "not_performed",
        },
    }
    with open(f"{ROOT}/reports/run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    return manifest


def build_md(manifest):
    m = manifest
    lines = ["# HabitDrive smoke-run configuration (machine + human readable)",
             "", f"- generated: {m['generated_at']}",
             f"- task: {m['task']}",
             "", "## Data", f"- input: `{m['data']['input_dir']}`",
             f"- version: {m['data']['version']}",
             f"- QC report: {m['data']['qc_report']}",
             f"- smoke manifest: {m['data']['smoke_manifest']} "
             f"(seed {m['data']['smoke_manifest_seed']})",
             "", "## Environment",
             f"- OS: {m['environment']['os']}",
             f"- Python: {m['environment']['python']}",
             f"- torch: {m['environment']['torch']} (CUDA {m['environment']['cuda']}; "
             f"available on login node: {m['environment']['cuda_available_on_login']})",
             "- GPU: NVIDIA A40 x4 on Slurm node gpu1 (job via sbatch)",
             "", "## Models", ]
    for k, v in m["models"].items():
        lines.append(f"### {k}")
        lines.append(f"- paper: {v['paper']}")
        lines.append(f"- used: {v['used']}  (**is_paper_config={v['is_paper_config']}**)")
        lines.append(f"- note: {v['note']}")
    lines += ["", "## Methods", "| method | status | n_tasks | exc | parse_fail | note |",
              "|---|---|---|---|---|---|"]
    for name, v in m["methods"].items():
        lines.append(f"| {name} | {v['status']} | {v['n_tasks']} | "
                     f"{v['n_exceptions']} | {v['n_parse_fail']} | "
                     f"{v['error'] or ''} |")
    lines += ["", "## Scoring", f"- entry: {m['scoring']['entry']}",
              f"- metrics: {m['scoring']['metrics']}",
              f"- denominators: {m['scoring']['denominator_rules']}",
              "", "## Error attribution",
              f"- entry: {m['attribution']['entry']}",
              f"- rule: {m['attribution']['rule']}",
              "", "## Independent review",
              f"- table: {m['review']['table']}",
              f"- agreement script: {m['review']['agreement_script']}",
              f"- human annotation status: {m['review']['human_annotation']}",
              "",
              "## Substitute-backend disclosure",
              "The paper-specified decision model, memory-construction model, and "
              "embedding model are NOT reachable from this server (the OpenAI-"
              "compatible gateway `xiaoai.plus` is unreachable, and no local "
              "embedding model / vLLM is installed). This run therefore uses "
              "local Llama-3.1-8B as the decision/memory backend and keyword "
              "matching as retrieval, and is a **substitute-backend smoke run**, "
              "not a paper-model reproduction.",
              ]
    with open(f"{ROOT}/reports/run_config.md", "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {ROOT}/reports/run_config.md and run_manifest.json")


if __name__ == "__main__":
    manifest = build_manifest()
    build_md(manifest)
