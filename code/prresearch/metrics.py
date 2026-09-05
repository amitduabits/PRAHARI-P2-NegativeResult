"""Metric helpers shared by all six papers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def topk_accuracy(ranked: Sequence[Sequence[str]], truth: Sequence[str], k: int) -> float:
    if not truth:
        return 0.0
    hits = sum(1 for cand, y in zip(ranked, truth) if y in list(cand)[:k])
    return hits / len(truth)


def percentile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def bootstrap_ci(values: Sequence[float], rng: np.random.Generator, reps: int = 2000, alpha: float = 0.05):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, arr.size, size=(reps, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def expected_calibration_error(pred: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        sel = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if not sel.any():
            continue
        total += sel.mean() * abs(p[sel].mean() - a[sel].mean())
    return float(total)
