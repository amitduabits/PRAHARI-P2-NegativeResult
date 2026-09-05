import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from prresearch.p2_fallback.engines import PAIRS, SinglePipeline, TwoTierPipeline, make_samples
from prresearch.p2_fallback.estimator import ProvenanceAccuracyEstimator


def test_record_shape_identical_across_tiers():
    mk_p, mk_s = PAIRS["anpr"]
    pipe = TwoTierPipeline(mk_p(), mk_s())
    recs = [pipe.run(s).as_dict() for s in make_samples(400, "text", "t:anpr")]
    keys = {frozenset(r) for r in recs}
    assert len(keys) == 1
    assert {r["path"] for r in recs} == {"primary", "secondary"}


def test_two_tier_never_drops_a_frame():
    mk_p, mk_s = PAIRS["objects"]
    samples = make_samples(500, "vehicle", "t:obj")
    two = [TwoTierPipeline(mk_p(), mk_s()).run(s) for s in samples]
    assert all(d is not None for d in two)
    single = SinglePipeline(mk_p())
    lost = sum(1 for s in samples if single.run(s) is None)
    assert lost > 0


def test_secondary_is_deterministic():
    _, mk_s = PAIRS["faces"]
    samples = make_samples(300, "person", "t:face")
    assert [mk_s()(s) for s in samples] == [mk_s()(s) for s in samples]


def test_estimator_separates_paths():
    mk_p, mk_s = PAIRS["objects"]
    pipe = TwoTierPipeline(mk_p(), mk_s())
    recs, correct = [], []
    for s in make_samples(4000, "vehicle", "t:est"):
        d = pipe.run(s)
        recs.append(d.as_dict())
        correct.append(d.value == s.truth)
    est = ProvenanceAccuracyEstimator().fit(recs[:1500], correct[:1500])
    prod, y = recs[1500:], correct[1500:]
    prim = [i for i, r in enumerate(prod) if r["path"] == "primary"]
    sec = [i for i, r in enumerate(prod) if r["path"] == "secondary"]
    per = est.predict_per_record(prod)
    assert per[prim].mean() > per[sec].mean()
    true_gap = sum(y[i] for i in prim) / len(prim) - sum(y[i] for i in sec) / len(sec)
    est_gap = per[prim].mean() - per[sec].mean()
    assert abs(true_gap - est_gap) < 0.10
