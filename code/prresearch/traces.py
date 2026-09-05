"""Synthetic camera estate and detection traces.

Grounded on the PRAHARI seeded registry (Gujarat corridor, Valsad to
Gandhinagar) but generated so that the whole programme is reproducible without
shipping operational footage. Three topologies are supported:

    grid        regular street grid, the case road-network baselines are good at
    smallworld  Watts-Strogatz rewiring, mixed regular and long-range links
    irregular   godowns, parks and border crossings: sparse, non-planar links,
                the case the strategy document claims road networks fail on
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from prresearch.seeds import rng

Topology = Literal["grid", "smallworld", "irregular"]

OWNERSHIPS = ("Own", "Gov", "Private", "Partner")
DATA_USE = ("public_safety", "traffic_only", "restricted")
JURISDICTIONS = ("GJ-VLS", "GJ-SRT", "GJ-VAD", "GJ-AHM", "GJ-GNR")


@dataclass(frozen=True)
class Camera:
    camera_id: str
    lat: float
    lon: float
    ownership: str
    cert_valid: bool
    data_use_class: str
    jurisdiction: str
    fov_depth_m: float
    speed_limit_kmh: float


@dataclass
class Estate:
    cameras: list[Camera]
    adjacency: dict[str, list[str]] = field(default_factory=dict)

    def by_id(self) -> dict[str, Camera]:
        return {c.camera_id: c for c in self.cameras}

    def index(self) -> dict[str, int]:
        return {c.camera_id: i for i, c in enumerate(self.cameras)}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Same formula as app/store.haversine_km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _grid_links(side: int) -> dict[int, list[int]]:
    links: dict[int, list[int]] = {i: [] for i in range(side * side)}
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if c + 1 < side:
                links[i].append(i + 1)
                links[i + 1].append(i)
            if r + 1 < side:
                links[i].append(i + side)
                links[i + side].append(i)
    return links


def build_estate(n: int, topology: Topology = "grid", seed_name: str = "estate") -> Estate:
    g = rng(f"{seed_name}:{topology}:{n}")
    side = int(math.ceil(math.sqrt(n)))
    cams: list[Camera] = []
    # Valsad (20.61, 72.93) to Gandhinagar (23.22, 72.65): a 320 km corridor.
    lat0, lon0 = 20.61, 72.93
    for i in range(n):
        r, c = divmod(i, side)
        jitter = g.normal(0.0, 0.0035, size=2)
        lat = lat0 + (r / max(side - 1, 1)) * 2.61 + float(jitter[0])
        lon = lon0 + (c / max(side - 1, 1)) * 0.55 + float(jitter[1])
        cams.append(
            Camera(
                camera_id=f"CAM{i:05d}",
                lat=round(lat, 6),
                lon=round(lon, 6),
                ownership=str(g.choice(OWNERSHIPS, p=[0.25, 0.40, 0.25, 0.10])),
                cert_valid=bool(g.random() < 0.88),
                data_use_class=str(g.choice(DATA_USE, p=[0.55, 0.30, 0.15])),
                jurisdiction=str(g.choice(JURISDICTIONS)),
                fov_depth_m=float(g.uniform(35.0, 110.0)),
                speed_limit_kmh=float(g.choice([40.0, 60.0, 80.0, 100.0])),
            )
        )

    raw = _grid_links(side)
    adjacency: dict[str, list[str]] = {}
    ids = [c.camera_id for c in cams]

    if topology == "grid":
        for i in range(n):
            adjacency[ids[i]] = [ids[j] for j in raw[i] if j < n]
    elif topology == "smallworld":
        for i in range(n):
            nbrs = [j for j in raw[i] if j < n]
            kept = [j for j in nbrs if g.random() > 0.15]
            extra = [int(g.integers(0, n)) for _ in range(int(g.integers(0, 3)))]
            adjacency[ids[i]] = sorted({ids[j] for j in kept + extra if j != i})
    else:  # irregular
        for i in range(n):
            nbrs = [j for j in raw[i] if j < n]
            kept = [j for j in nbrs if g.random() > 0.55]
            hops = int(g.integers(1, 5))
            extra = [int(g.integers(0, n)) for _ in range(hops)]
            adjacency[ids[i]] = sorted({ids[j] for j in kept + extra if j != i})
    for cid in ids:
        adjacency.setdefault(cid, [])
    return Estate(cameras=cams, adjacency=adjacency)


def generate_trips(
    estate: Estate,
    n_vehicles: int,
    steps: tuple[int, int] = (4, 14),
    seed_name: str = "trips",
) -> list[list[tuple[str, float]]]:
    """Random walks over the true adjacency, biased so a few corridors dominate.

    Returns one list per vehicle of (camera_id, timestamp_seconds). The walker
    never sees the adjacency: only these sequences reach the predictors.
    """
    g = rng(seed_name)
    ids = [c.camera_id for c in estate.cameras]
    by_id = estate.by_id()
    trips: list[list[tuple[str, float]]] = []
    # Corridor bias: each node prefers one outgoing edge, so transition
    # frequencies carry signal instead of being uniform.
    preferred = {cid: (nbrs[int(g.integers(0, len(nbrs)))] if nbrs else cid)
                 for cid, nbrs in estate.adjacency.items()}
    for _ in range(n_vehicles):
        cur = ids[int(g.integers(0, len(ids)))]
        t = float(g.uniform(0, 86400))
        n_steps = int(g.integers(steps[0], steps[1]))
        trip = [(cur, t)]
        for _ in range(n_steps):
            nbrs = estate.adjacency.get(cur) or []
            if not nbrs:
                break
            if g.random() < 0.62:
                nxt = preferred[cur]
            else:
                nxt = nbrs[int(g.integers(0, len(nbrs)))]
            a, b = by_id[cur], by_id[nxt]
            km = haversine_km(a.lat, a.lon, b.lat, b.lon)
            speed = max(a.speed_limit_kmh * float(g.uniform(0.55, 1.05)), 5.0)
            t += (km / speed) * 3600.0 + float(g.uniform(5, 60))
            trip.append((nxt, t))
            cur = nxt
        if len(trip) >= 2:
            trips.append(trip)
    return trips


def plate(i: int) -> str:
    """Gujarat plate shape, matching the seeded registry (GJ01AB1234)."""
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return f"GJ{(i % 38) + 1:02d}{letters[(i // 38) % 24]}{letters[(i // 912) % 24]}{i % 10000:04d}"
