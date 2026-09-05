"""Label-free accuracy estimation from inference provenance.

The paper's claim: because every record says which tier produced it and with
what confidence, an operator can estimate the accuracy of a batch of detections
without any ground-truth labels for that batch.

Method. Calibrate once on a small labelled reference set: for each (path,
entity_type) stratum, fit a piecewise-constant map from reported confidence to
empirical correctness over equal-count bins. Then, for unlabelled production
records, the estimated accuracy is the mean of the mapped values. This is a
stratified plug-in estimator; it is unbiased when the production confidence
distribution is covered by the calibration bins.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Stratum:
    edges: np.ndarray
    values: np.ndarray
    fallback: float

    def apply(self, conf: np.ndarray) -> np.ndarray:
        if self.values.size == 0:
            return np.full(conf.shape, self.fallback)
        idx = np.clip(np.searchsorted(self.edges, conf, side="right") - 1, 0, self.values.size - 1)
        return self.values[idx]


class ProvenanceAccuracyEstimator:
    def __init__(self, bins: int = 8) -> None:
        self.bins = bins
        self.strata: dict[tuple[str, str], Stratum] = {}
        self.global_rate = 0.5

    def fit(self, records: Sequence[dict], correct: Sequence[bool]) -> "ProvenanceAccuracyEstimator":
        y = np.asarray(correct, dtype=float)
        self.global_rate = float(y.mean()) if y.size else 0.5
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, r in enumerate(records):
            groups[(r["path"], r["entity_type"])].append(i)
        for key, idxs in groups.items():
            conf = np.asarray([records[i]["confidence"] for i in idxs], dtype=float)
            yy = y[idxs]
            order = np.argsort(conf)
            conf, yy = conf[order], yy[order]
            nb = max(1, min(self.bins, conf.size // 20 or 1))
            splits = np.array_split(np.arange(conf.size), nb)
            edges, values = [], []
            for s in splits:
                if s.size == 0:
                    continue
                edges.append(conf[s[0]])
                values.append(float(yy[s].mean()))
            self.strata[key] = Stratum(
                edges=np.asarray(edges, dtype=float),
                values=np.asarray(values, dtype=float),
                fallback=float(yy.mean()) if yy.size else self.global_rate,
            )
        return self

    def predict_per_record(self, records: Sequence[dict]) -> np.ndarray:
        out = np.empty(len(records), dtype=float)
        for i, r in enumerate(records):
            st = self.strata.get((r["path"], r["entity_type"]))
            if st is None:
                out[i] = self.global_rate
            else:
                out[i] = float(st.apply(np.asarray([r["confidence"]]))[0])
        return out

    def estimate_batch_accuracy(self, records: Sequence[dict]) -> float:
        if not records:
            return float("nan")
        return float(self.predict_per_record(records).mean())


class ConfidenceOnlyEstimator:
    """Baseline: ignore provenance, treat reported confidence as the probability."""

    def fit(self, records, correct):  # noqa: ARG002 - signature parity
        return self

    def estimate_batch_accuracy(self, records: Sequence[dict]) -> float:
        if not records:
            return float("nan")
        return float(np.mean([r["confidence"] for r in records]))


class GlobalPriorEstimator:
    """Baseline: one number, the accuracy measured on the calibration set."""

    def __init__(self) -> None:
        self.rate = 0.5

    def fit(self, records, correct):  # noqa: ARG002
        y = np.asarray(correct, dtype=float)
        self.rate = float(y.mean()) if y.size else 0.5
        return self

    def estimate_batch_accuracy(self, records: Sequence[dict]) -> float:
        return self.rate
