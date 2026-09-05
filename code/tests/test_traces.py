import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from prresearch.seeds import rng
from prresearch.traces import build_estate, generate_trips, haversine_km


def test_estate_is_reproducible():
    a = build_estate(120, "smallworld", seed_name="tt")
    b = build_estate(120, "smallworld", seed_name="tt")
    assert a.cameras == b.cameras
    assert a.adjacency == b.adjacency


def test_trips_follow_true_adjacency():
    e = build_estate(120, "grid", seed_name="tt")
    for trip in generate_trips(e, 200, seed_name="tt:trips")[:50]:
        for (c1, _), (c2, _) in zip(trip, trip[1:]):
            assert c2 in e.adjacency[c1]


def test_trip_timestamps_increase():
    e = build_estate(80, "grid", seed_name="tt")
    for trip in generate_trips(e, 100, seed_name="tt:t")[:30]:
        ts = [t for _, t in trip]
        assert ts == sorted(ts)


def test_haversine_zero_and_symmetric():
    assert haversine_km(20.0, 72.0, 20.0, 72.0) == 0.0
    assert abs(haversine_km(20, 72, 23, 73) - haversine_km(23, 73, 20, 72)) < 1e-9


def test_rng_is_deterministic():
    assert list(rng("x").random(5)) == list(rng("x").random(5))
