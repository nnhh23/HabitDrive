"""Generate the concise final overview (QC + methods + results + artifacts)."""
from __future__ import annotations

import json

ROOT = "/path/to/habitdrive_eval"
OUT = f"{ROOT}/data/outputs"
DATA = "/path/to/habitdrive_full/data"


def main():
    summ = json.load(open(f"{OUT}/summary.json", encoding="utf-8"))
    manifest = summ["manifest"]
    rep = json.load(open(f"{ROOT}/reports/representative_status.json",
                         encoding="utf-8"))
    att = json.load(open(f"{OUT}/attribution.json", encoding="utf-8"))

    lines = ["# HabitDrive smoke run — concise overview", ""]

    lines += ["## 1. Data QC conclusion",
              f"- QC report: `{ROOT}/reports/qc_report.md`",
              "- Full-data programmatic checks: **0 issues** on v2 (fixed).",
              "- v1 had 1 bug (cabin initial-state mutual exclusion, 924/1000 "
              "tasks); fixed in v2; v1 preserved at `.../habitdrive_full/data_v1`.",
              "- Pending human review: seeded NL/task samples are listed in the "
              "QC report (no human annotation performed).", ""]

    lines += ["## 2. Method status (11 methods)",
              "| method | status | backend |", "|---|---|---|"]
    for m in summ["summary"]:
        st = m["status"]
        if m["method"] in ("no_memory", "oracle_memory", "full_context",
                           "recursive_summarization", "vector_rag",
                           "key_value_store"):
            backend = "local (Llama-3.1-8B decision + keyword retrieval)"
        else:
            r = rep.get(m["method"], {})
            backend = r.get("blocked_reason", "")
            st = r.get("status", st)
        lines.append(f"| {m['method']} | {st} | {backend} |")
    lines += ["", "Basic strategies (6) ran end-to-end; representative systems "
              "(5) are **installed** but **not e2e-run** because the paper "
              "extraction/embedding backend is unreachable and no local "
              "embedding model is installed (substitute-backend smoke run).", ""]

    lines += ["## 3. Small-batch results (basic methods)",
              "| method | route E2E | DA | PSA | cabin E2E | TFA |",
              "|---|---|---|---|---|---|"]
    for m in summ["summary"]:
        if m["status"] != "e2e_run":
            continue
        met = m.get("metrics") or {}
        r = met.get("route") or {}
        c = met.get("cabin") or {}
        def pct(x):
            return f"{x*100:.1f}%" if x is not None else "n/a"
        lines.append(f"| {m['method']} | {pct(r.get('E2E'))} | {pct(r.get('DA'))} "
                     f"| {pct(r.get('PSA'))} | {pct(c.get('E2E'))} | "
                     f"{pct(c.get('TFA'))} |")
    lines += ["",
              "Small sample (36 tasks, 4 households) only confirms the pipeline "
              "works; it is NOT used to infer paper method rankings.", ""]

    lines += ["## 4. Artifacts (absolute paths)",
              f"- code: `{ROOT}/code/`",
              f"- config (human): `{ROOT}/reports/run_config.md`",
              f"- config (machine): `{ROOT}/reports/run_manifest.json`",
              f"- QC report: `{ROOT}/reports/qc_report.md`",
              f"- smoke manifest: `{ROOT}/data/smoke/manifest.json`",
              f"- per-task detail + raw outputs: `{ROOT}/data/outputs/<method>.json`",
              f"- scoring detail summary: `{ROOT}/data/outputs/summary.json`",
              f"- error attribution: `{ROOT}/data/outputs/attribution.json`",
              f"- independent review table: `{ROOT}/reports/review_table.csv`",
              f"- agreement script: `{ROOT}/code/agreement.py`",
              f"- logs: `{ROOT}/logs/`", ""]

    lines += ["## 5. Reproduction commands",
              "```bash",
              "conda activate habitdrive",
              f"cd {ROOT}/code",
              "sbatch smoke.slurm               # re-run the smoke (calls the model)",
              "python attribution.py            # recompute error attribution",
              "python review_table.py           # regenerate review table",
              "python report.py                 # regenerate config/manifest",
              "```",
              "Re-score from saved outputs (no model calls):",
              "```bash",
              f"cd {ROOT}/code",
              "python scoring.py                 # scoring self-test",
              "# metrics are recomputed inside smoke.py from saved per-task detail;",
              "# a no-model re-score entry point is provided by attribution.py +",
              "# review_table.py, which read data/outputs/*.json only.",
              "```",
              ]
    with open(f"{ROOT}/reports/FINAL_SUMMARY.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
