"""Paper 2 experiments.

E2.1  Yield and accuracy of the two-tier pipeline vs. primary-only and retry,
      across three engine pairs and a sweep of injected failure rates.
E2.2  Determinism of the secondary path: repeated runs are byte-identical.
E2.3  Label-free batch-accuracy estimation from provenance, against a
      confidence-only baseline and a global-prior baseline.
E2.4  The accuracy/latency trade of the secondary tier.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prresearch.metrics import bootstrap_ci, expected_calibration_error
from prresearch.p2_fallback.engines import (
    PAIRS,
    PrimaryEngine,
    RetryPipeline,
    SinglePipeline,
    TwoTierPipeline,
    make_samples,
)
from prresearch.p2_fallback.estimator import (
    ConfidenceOnlyEstimator,
    GlobalPriorEstimator,
    ProvenanceAccuracyEstimator,
)
from prresearch.seeds import rng

RESULTS = Path(__file__).resolve().parents[2] / "results"

# Measured on the deployed stack (02_Code/prahari), CPU only, per frame.
LATENCY_MS = {
    "yolo": 41.0, "opencv_blob": 6.2,
    "facenet": 55.0, "histogram": 3.1,
    "paddleocr": 78.0, "tesseract": 22.0,
}


def _run_pair(pair: str, n: int, fail_rate: float | None) -> dict:
    mk_p, mk_s = PAIRS[pair]
    cls = {"objects": "vehicle", "faces": "person", "anpr": "text"}[pair]
    samples = make_samples(n, cls, seed_name=f"p2:samples:{pair}")

    def primary() -> PrimaryEngine:
        p = mk_p()
        if fail_rate is not None:
            p.fail_rate = fail_rate
        return p

    two = TwoTierPipeline(primary(), mk_s())
    recs, correct = [], []
    for s in samples:
        d = two.run(s)
        recs.append(d.as_dict())
        correct.append(d.value == s.truth)
    n_secondary = sum(1 for r in recs if r["path"] == "secondary")

    single = SinglePipeline(primary())
    s_out = [single.run(s) for s in samples]
    s_hits = sum(1 for d, s in zip(s_out, samples) if d is not None and d.value == s.truth)

    retry = RetryPipeline(primary())
    r_out = [retry.run(s) for s in samples]
    r_hits = sum(1 for d, s in zip(r_out, samples) if d is not None and d.value == s.truth)

    return {
        "pair": pair,
        "injected_fail_rate": fail_rate,
        "n": n,
        "two_tier_yield": len(recs) / n,
        "two_tier_accuracy": float(np.mean(correct)),
        "secondary_share": n_secondary / n,
        "primary_only_yield": sum(1 for d in s_out if d is not None) / n,
        "primary_only_accuracy_over_all": s_hits / n,
        "retry_yield": sum(1 for d in r_out if d is not None) / n,
        "retry_accuracy_over_all": r_hits / n,
        "_records": recs,
        "_correct": correct,
    }


def e2_1_yield(n: int = 50000) -> dict:
    rows = []
    for pair in PAIRS:
        for fr in (0.05, 0.10, 0.20, 0.35, 0.50):
            r = _run_pair(pair, n // 5, fr)
            r.pop("_records"), r.pop("_correct")
            rows.append(r)
    return {"experiment": "E2.1_yield_accuracy_vs_failure", "rows": rows}


def e2_2_determinism(n: int = 5000) -> dict:
    rows = []
    for pair in PAIRS:
        _, mk_s = PAIRS[pair]
        cls = {"objects": "vehicle", "faces": "person", "anpr": "text"}[pair]
        samples = make_samples(n, cls, seed_name=f"p2:samples:{pair}")
        a = [mk_s()(s) for s in samples]
        b = [mk_s()(s) for s in samples]
        rows.append(
            {
                "pair": pair,
                "secondary_runs_identical": a == b,
                "mismatches": sum(1 for x, y in zip(a, b) if x != y),
                "n": n,
            }
        )
    return {"experiment": "E2.2_secondary_determinism", "rows": rows}


def e2_3_label_free(n: int = 60000, calib_frac: float = 0.15) -> dict:
    g = rng("p2:split")
    rows = []
    for pair in PAIRS:
        run = _run_pair(pair, n // 3, None)
        recs, correct = run["_records"], run["_correct"]
        idx = g.permutation(len(recs))
        k = int(len(recs) * calib_frac)
        cal, prod = idx[:k], idx[k:]
        cal_r = [recs[i] for i in cal]
        cal_y = [correct[i] for i in cal]
        prod_r = [recs[i] for i in prod]
        prod_y = np.asarray([correct[i] for i in prod], dtype=float)
        truth = float(prod_y.mean())
        entry = {"pair": pair, "true_accuracy": truth, "n_calibration": k, "n_production": len(prod)}
        for name, est in (
            ("provenance_stratified", ProvenanceAccuracyEstimator().fit(cal_r, cal_y)),
            ("confidence_only", ConfidenceOnlyEstimator().fit(cal_r, cal_y)),
            ("global_prior", GlobalPriorEstimator().fit(cal_r, cal_y)),
        ):
            pred = est.estimate_batch_accuracy(prod_r)
            entry[f"{name}_estimate"] = pred
            entry[f"{name}_abs_error"] = abs(pred - truth)
        # Per-record calibration quality of the proposed estimator.
        pe = ProvenanceAccuracyEstimator().fit(cal_r, cal_y)
        per = pe.predict_per_record(prod_r)
        entry["provenance_ece"] = expected_calibration_error(per, prod_y)
        lo, hi = bootstrap_ci(list(prod_y), rng(f"p2:boot:{pair}"), reps=500)
        entry["true_accuracy_ci95"] = [lo, hi]
        # Sliced by inference path, the number an operator actually wants.
        for path in ("primary", "secondary"):
            sel = [i for i, r in enumerate(prod_r) if r["path"] == path]
            if sel:
                entry[f"true_accuracy_{path}"] = float(prod_y[sel].mean())
                entry[f"estimate_{path}"] = float(per[sel].mean())
        rows.append(entry)
    return {"experiment": "E2.3_label_free_accuracy_estimation", "rows": rows}


def e2_4_latency(n: int = 20000) -> dict:
    rows = []
    for pair in PAIRS:
        run = _run_pair(pair, n // 3, None)
        recs = run["_records"]
        lat = [LATENCY_MS[r["engine"]] if r["path"] == "primary"
               else LATENCY_MS[list(PAIRS[pair])[0]().name] + LATENCY_MS[r["engine"]]
               for r in recs]
        rows.append(
            {
                "pair": pair,
                "mean_latency_ms": float(np.mean(lat)),
                "p99_latency_ms": float(np.percentile(lat, 99)),
                "secondary_share": run["secondary_share"],
                "accuracy": run["two_tier_accuracy"],
            }
        )
    return {"experiment": "E2.4_latency_of_fallback", "rows": rows}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {
        "paper": "P2 Deterministic fallback engines and reproducible inference",
        "results": [e2_1_yield(), e2_2_determinism(), e2_3_label_free(), e2_4_latency()],
    }
    path = RESULTS / "p2_fallback.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    for block in out["results"]:
        print("\n#", block["experiment"])
        for row in block["rows"]:
            print(" ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
