"""pass@k for every k the sample count supports, plus the health counters.

Reporting one k hides the shape of the result. A model that solves a problem
once in sixteen tries and one that solves it every time both score pass@16 = 1;
the whole ladder from pass@1 up separates them, and it is free - the same n
samples produce every k.

pass@k uses the unbiased estimator of Chen et al. (2021):

    pass@k = 1 - C(n-c, k) / C(n, k)

which is the probability that a random k-subset of the n drawn samples contains
at least one correct answer. Averaging that over problems is what makes pass@4
from 8 samples an estimate of "how often would 4 tries succeed", rather than a
number that only means something at k = n.
"""
from __future__ import annotations

import math


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem: c correct out of n samples."""
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} samples, got {n}")
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


def ladder(n: int) -> list[int]:
    """The k values worth reporting for n samples: 1, 2, 4, 8, ... n.

    Powers of two plus n itself. Every k from 1 to n is computable, but 16 of
    them for n=16 is a wall of numbers whose neighbours differ by noise.
    """
    ks, k = [], 1
    while k <= n:
        ks.append(k)
        k *= 2
    if ks[-1] != n:
        ks.append(n)
    return ks


def summarize(records: list[dict], extra: dict | None = None) -> dict:
    """Aggregate per-problem records into the full metric set.

    Each record needs `n` (samples) and `c` (correct), and `samples` carrying
    per-sample `truncated` / `extracted` for the health counters.
    """
    if not records:
        raise ValueError("no records to summarize")
    ns = {r["n"] for r in records}
    if len(ns) != 1:
        raise ValueError(f"records disagree on sample count: {sorted(ns)}")
    n = ns.pop()
    N = len(records)

    out: dict = {"n_problems": N, "samples_per_problem": n}
    for k in ladder(n):
        vals = [pass_at_k(r["n"], r["c"], k) for r in records]
        mean = sum(vals) / N
        var = sum((v - mean) ** 2 for v in vals) / max(N - 1, 1)
        out[f"pass@{k}"] = mean
        out[f"pass@{k}_stderr"] = math.sqrt(var / N)

    # Health counters. A high truncation or no-answer rate means the number
    # above is a harness artefact, not a model result, so they travel with it.
    samples = [s for r in records for s in r["samples"]]
    total = len(samples) or 1
    out["truncation_rate"] = sum(bool(s.get("truncated")) for s in samples) / total
    out["no_answer_rate"] = sum(s.get("extracted") is None for s in samples) / total
    out["mean_gen_tokens"] = sum(s.get("n_tokens", 0) for s in samples) / total
    out.update(extra or {})
    return out
