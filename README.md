# HabitDrive

HabitDrive is a benchmark for **long-term personalized memory in multi-user
in-vehicle agents**. It organizes diverse households, specifies each member's
preferences and how they evolve over time, and converts those structured
trajectories into chronological, natural-language interaction histories.
Evaluation requests are then constructed from the preferences that are valid
at the end of each history, and agents must select personalized routes or
adjust cabin settings through simulated tool calls.

This repository contains the **full dataset** and the **generation +
evaluation pipeline** used to build and validate it.

## Dataset

| Component | Count |
|---|---|
| Households | 100 |
| Members | 263 |
| Structured preference records | 3,200 |
| Base history events (establishment + evolution) | 5,000 |
| Expanded history interactions (total) | ~8,277 |
| Route-planning tasks | 2,000 (a subset require parking) |
| In-cabin setting tasks | 1,000 |

Four (overlapping) memory-challenge labels are defined: **user binding**,
**preference updates**, **conditional application**, and **temporary-request
interference**. Route tasks are staged (destination → route → parking when
required); in-cabin tasks cover five device categories (climate, seats,
media, windows, ambient lighting) with target fields, protected fields, and
setting constraints.

## Repository layout

- `data/` — the dataset:
  - `households.json`, `memories.json`, `structured_history.json`,
    `natural_language_history.json`
  - `route_tasks.json`, `cabin_tasks.json` (each task has a `public` section
    for model input and an `evaluation` section with the reference answer,
    supporting evidence, and challenge labels)
  - `route_tasks_public.json`, `cabin_tasks_public.json` (public-only views)
  - `manifest.json`
- `generation/` — deterministic dataset generation (households → members →
  preferences → evolution trajectories → history → tasks), local-model
  verbalization, and validation.
- `evaluation/` — the memory-method pipeline, decision model, scoring
  (E2E / DA / PSA / TFA), error attribution, and review-table export.
- `docs/` — the data-quality report and generation notes.

## Reproducing the dataset

Requirements: Python 3.11, PyTorch 2.5+ (CUDA), `transformers`, and a local
instruct model for verbalization (e.g. Llama-3.1-8B-Instruct).

```bash
cd generation
python run.py structure        # deterministic structure + validation (CPU)
sbatch verbalize_all.slurm     # natural-language verbalization on GPU
python finalize.py             # aggregate history + manifest
python validate.py ../data     # validate from disk
```

`GENERATION_NOTES.md` records every decision for values the paper leaves
unspecified, kept consistent across the whole dataset.

## Reproducing the evaluation

```bash
cd evaluation
python scoring.py              # scoring self-test (E2E/DA/PSA/TFA)
# memory methods are defined in methods.py; scoring in scoring.py;
# error attribution in attribution.py; review-table export in review_table.py
```

The memory-construction, embedding, and decision models are configured in
`evaluation/decision.py` and `evaluation/methods.py` and can be pointed at
the paper's backends (DeepSeek V4 Flash / Gemini 3.5 Flash-Lite for decision,
Qwen3-32B for memory construction, BAAI/bge-m3 for embedding).

## License

MIT (see `LICENSE`).
