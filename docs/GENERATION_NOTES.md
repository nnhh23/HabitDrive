# HabitDrive Dataset — Generation Notes

This document records the decisions I made where the paper did not specify
exact values or a unique implementation. Everything the paper states is
implemented exactly as written. Nothing here is an evaluation claim; it is a
record of construction choices kept consistent across the whole dataset.

## Paper-fixed requirements (implemented verbatim)

- **100 households**, each an independent memory namespace (`namespace_id ==
  household_id`).
- **5,000 base history events** (preference establishment + evolution), i.e.
  50 per household, before repeated applications / temporary requests /
  unrelated exchanges are added.
- **2,000 route-planning tasks** and **1,000 in-cabin tasks**.
- Member attributes: id, household role, occupation, age group, driving
  eligibility.
- Six evolution types: no_change, content_update, condition_update,
  unconditional_to_conditional, conditional_to_unconditional, cancellation.
- Four (overlapping) challenge labels: user_binding, preference_update,
  conditional_application, temporary_request_interference.
- Route tasks are staged: destination resolution → route selection → parking
  selection (parking only when required).
- In-cabin settings cover five device categories: climate, seats, media,
  windows, ambient lighting.

## Decisions (paper-unspecified; kept consistent)

1. **Household composition distribution** (paper says "diverse"):
   two_adults_one_child 0.40, two_adults 0.30, one_adult_one_child 0.20,
   two_adults_two_children 0.10. All households are multi-user (2–4 members)
   so the user-binding challenge is well represented. → 263 members total.

2. **Per-household task split**: exactly 20 route tasks and 10 cabin tasks per
   household (100 × 20 = 2,000; 100 × 10 = 1,000). This is the simplest way to
   hit the paper's exact totals deterministically.

3. **Base-event composition**: each household gets exactly 50 base events =
   establishments + evolutions. Establishments are generated naturally
   (locations, route preferences, parking preferences, passenger constraints,
   in-cabin preferences with a floor of 32 so the evolution budget stays
   reasonable); the remaining slots are filled with evolution steps applied
   one-at-a-time to an active pool of preferences (home is never evolved).

4. **Parking subset**: a route task requires parking only when the driver has
   an applicable parking preference for the destination and a seeded coin
   passes (~40% target). The actual count is reported (391), not asserted as a
   paper target.

5. **Default route objective**: when no personal route preference applies, the
   task uses "minimize travel time" as the default navigation objective so a
   unique gold route always exists (mirrors the "default objective" fallback).

6. **Conflict resolution**: when several same-attribute preferences would
   otherwise conflict (e.g. two different preferred parking lots, or "avoid
   highway" vs "prefer highway"), the most recently updated memory wins. This
   keeps every candidate set to exactly one gold.

7. **Candidates**: 3 route candidates and 3 parking candidates per stage,
   generated deterministically so exactly one satisfies all applicable
   requirements; the other two violate distinct requirements. The gold is not
   forced to be best on every attribute (e.g. it is not always the fastest).

8. **Language**: history is verbalized by the local Llama-3.1-8B-Instruct
   model (grounded in the structured facts; template fallback on failure).
   Task request text uses deterministic templates (so request→gold mapping is
   exact). Unrelated exchanges are already natural and are kept as templates.

9. **Expanded history size** (paper only fixes base events): 20–30 recurring
   application events, 1–2 one-trip temporary instructions, and 5–8 unrelated
   exchanges per household (our choice; actual totals reported).

10. **Reference answers / evidence**: gold destination, gold route id, gold
    parking id (null when parking not required), plus the supporting memory
    ids and challenge labels, are stored under `evaluation` and are never part
    of the model-visible `public` input. Public-only task files are also
    written.

11. **Time horizon**: creation Jan–Jun 2025, evolution through Nov 2025,
    history ends Dec 20 2025; tasks are timestamped after the history ends.

## Reproducibility

- Master seed 42; per-household/per-memory RNG streams are derived
  deterministically from the seed + entity id.
- Structured generation is fully deterministic and CPU-only.
- Verbalization is the only model-dependent step; it does not change gold.
