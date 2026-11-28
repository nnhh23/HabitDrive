# HabitDrive data QC report

- data dir: `/path/to/habitdrive_full/data`
- sample seed: 20251126
- total issues: 0

## Counts (paper target / actual)

| metric | paper | actual |
|---|---|---|
| households | 100 | 100 |
| base_events | 5000 | 5000 |
| route_tasks | 2000 | 2000 |
| cabin_tasks | 1000 | 1000 |
| members | (unspecified) | 263 |
| memories | (unspecified) | 3200 |
| expanded events | (unspecified) | 3275 |
| parking subset | (unspecified) | 417 |

## Challenge label counts (multi-label; sum != task total)
| label | count |
|---|---|
| conditional_application | 1277 |
| preference_update | 1878 |
| temporary_request_interference | 306 |
| user_binding | 2344 |

## Checks
| check | value |
|---|---|
| route_gold_mismatch | 0 |
| route_unique_fail | 0 |
| cabin_target_bad | 0 |
| cabin_constraint_bad | 0 |
| nl_fact_preservation_fail | 368 |

## Note on independence
Reference-answer validity re-uses the generator's requirement resolver (disclosed); all other checks are independent.

## Issue list (first 50)
```
```
### v3 (fixed route condition vocabulary, at `.../habitdrive_full/data`)
- **Bug found**: route preference conditions used a *different vocabulary* than
  task contexts (weather `rain/fog/snow/icy` vs task `rainy/snowy/foggy`;
  time `morning_rush/evening_rush` vs task `morning/evening`; trip context
  `in_a_hurry` vs task `commute/daily_errand`), so many route conditional
  preferences never activated — weakening the conditional-application
  challenge. (Cabin conditions already used the task vocabulary, so only route
  conditions were affected.)
- **Fix** (`constants.py`, `generate_structure.py`): align route condition
  values with task-context values.
- **Effect**: conditional-application task instances rose from 1,008 → 1,277;
  parking subset 391 → 417. Re-validation → **0 issues**.
- v2 is preserved at `.../habitdrive_full/data_v2`; the smoke run uses **v3**.
