"""Inter-rater agreement for the review table (run only after two reviewers
have filled `reviewer_judgement` in two separate copies).

This computes Cohen's kappa and raw agreement over the reviewer_judgement
column. It is a prepared workflow only; no real human annotations exist, so
no kappa is reported anywhere in the run summary.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter


def cohens_kappa(a, b):
    from math import isclose
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return None
    n_agree = sum(1 for x, y in zip(a, b) if x == y)
    po = n_agree / n
    # expected agreement by chance
    pe = 0.0
    for lab in labels:
        pa = sum(1 for x in a if x == lab) / n
        pb = sum(1 for x in b if x == lab) / n
        pe += pa * pb
    if isclose(pe, 1.0):
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["review_id"]: r["reviewer_judgement"].strip() for r in rows}


def main(a_path, b_path):
    a = load(a_path)
    b = load(b_path)
    common = sorted(set(a) & set(b))
    print(f"review_ids in both files: {len(common)}")
    a_vals = [a[k] for k in common]
    b_vals = [b[k] for k in common]
    agree = sum(1 for x, y in zip(a_vals, b_vals) if x == y)
    print(f"raw agreement: {agree}/{len(common)} "
          f"({agree/len(common)*100:.1f}%)" if common else "no common rows")
    print(f"Cohen's kappa: {cohens_kappa(a_vals, b_vals)}")
    print("label distribution A:", dict(Counter(a_vals)))
    print("label distribution B:", dict(Counter(b_vals)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python agreement.py review_a.csv review_b.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
