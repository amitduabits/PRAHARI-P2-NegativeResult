"""Paper 2: two-tier engine pairs with per-detection provenance.

Mirrors the fallback wiring in app/services/objects.py (YOLO backend, blob
fallback) and app/services/anpr.py (PaddleOCR primary, Tesseract secondary).
The primary is stochastic and may raise, return nothing, or return a wrong
answer. The secondary is a different algorithm class: lower accuracy, but
deterministic given the same input.

Every emitted record carries the same nine fields regardless of which tier
produced it, plus `engine` and `path`. That is the property the paper calls
engine equivalence under degradation.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from prresearch.seeds import rng


@dataclass(frozen=True)
class Sample:
    """One frame's worth of input, with the label held out of the pipeline."""

    sample_id: int
    difficulty: float          # 0 easy, 1 hard; drives both engines
    entity_class: str          # vehicle | person | text
    truth: str                 # never visible to the engines or the estimator


@dataclass
class Detection:
    """The nine-field record shape, identical for both tiers."""

    event_id: str
    entity_type: str
    entity_id: str
    value: str
    confidence: float
    camera_id: str
    ts: float
    engine: str
    path: str                  # "primary" | "secondary" | "none"

    def as_dict(self) -> dict:
        return asdict(self)


class PrimaryEngine:
    """Learned model: accurate when the input is easy, unreliable otherwise."""

    def __init__(self, name: str, skill: float, fail_rate: float, seed_name: str) -> None:
        self.name = name
        self.skill = skill
        self.fail_rate = fail_rate
        self._g = rng(seed_name)
        self.calls = 0

    def __call__(self, s: Sample) -> tuple[str | None, float]:
        self.calls += 1
        u = float(self._g.random())
        if u < self.fail_rate * (0.5 + s.difficulty):
            raise RuntimeError(f"{self.name}: inference exception")
        if float(self._g.random()) < 0.06 * s.difficulty:
            return None, 0.0                      # empty result, not an exception
        p_correct = float(np.clip(self.skill * (1.0 - 0.85 * s.difficulty), 0.02, 0.995))
        correct = float(self._g.random()) < p_correct
        value = s.truth if correct else s.truth[::-1] + "X"
        # Confidence is informative but imperfectly calibrated, as in deployment.
        conf = float(np.clip(p_correct + self._g.normal(0.0, 0.08), 0.01, 0.999))
        return value, conf


class SecondaryEngine:
    """Deterministic classical engine: OpenCV blob, histogram, Tesseract.

    No randomness at all. The same sample always produces the same output, which
    is what makes the fallback path reproducible and auditable.
    """

    def __init__(self, name: str, skill: float) -> None:
        self.name = name
        self.skill = skill
        self.calls = 0

    def __call__(self, s: Sample) -> tuple[str | None, float]:
        self.calls += 1
        # Deterministic decision rule: a stable hash of the sample stands in for
        # the pixel-level threshold the classical engine applies. Python's
        # built-in hash() is salted per process, so it cannot be used here
        # without breaking cross-run reproducibility.
        h = (zlib.crc32(f"{s.sample_id}:{s.entity_class}".encode()) & 0xFFFF) / 0xFFFF
        p_correct = float(np.clip(self.skill * (1.0 - 0.95 * s.difficulty), 0.01, 0.95))
        correct = h < p_correct
        value = s.truth if correct else s.truth[:-1] + "0"
        return value, round(p_correct, 4)


class TwoTierPipeline:
    """Primary, then deterministic secondary on exception, absence or empty result."""

    def __init__(self, primary: PrimaryEngine, secondary: SecondaryEngine) -> None:
        self.primary = primary
        self.secondary = secondary

    def run(self, s: Sample, camera_id: str = "CAM00000") -> Detection:
        value: str | None
        try:
            value, conf = self.primary(s)
            path = "primary"
            engine = self.primary.name
        except Exception:
            value, conf, path, engine = None, 0.0, "secondary", self.secondary.name
        if value is None:
            value, conf = self.secondary(s)
            path, engine = "secondary", self.secondary.name
        return Detection(
            event_id=f"{camera_id}:{s.sample_id}",
            entity_type=s.entity_class,
            entity_id=value or "",
            value=value or "",
            confidence=float(conf),
            camera_id=camera_id,
            ts=float(s.sample_id),
            engine=engine,
            path=path,
        )


class SinglePipeline:
    """Baseline: primary only. On failure the detection is simply lost."""

    def __init__(self, primary: PrimaryEngine) -> None:
        self.primary = primary

    def run(self, s: Sample, camera_id: str = "CAM00000") -> Detection | None:
        try:
            value, conf = self.primary(s)
        except Exception:
            return None
        if value is None:
            return None
        return Detection(
            event_id=f"{camera_id}:{s.sample_id}",
            entity_type=s.entity_class,
            entity_id=value,
            value=value,
            confidence=float(conf),
            camera_id=camera_id,
            ts=float(s.sample_id),
            engine=self.primary.name,
            path="primary",
        )


class RetryPipeline:
    """Baseline: on failure, call the same engine again (`try again`)."""

    def __init__(self, primary: PrimaryEngine, attempts: int = 3) -> None:
        self.primary = primary
        self.attempts = attempts

    def run(self, s: Sample, camera_id: str = "CAM00000") -> Detection | None:
        for _ in range(self.attempts):
            try:
                value, conf = self.primary(s)
            except Exception:
                continue
            if value is None:
                continue
            return Detection(
                event_id=f"{camera_id}:{s.sample_id}",
                entity_type=s.entity_class,
                entity_id=value,
                value=value,
                confidence=float(conf),
                camera_id=camera_id,
                ts=float(s.sample_id),
                engine=self.primary.name,
                path="primary",
            )
        return None


PAIRS: dict[str, tuple[Callable[[], PrimaryEngine], Callable[[], SecondaryEngine]]] = {
    "objects": (lambda: PrimaryEngine("yolo", skill=0.94, fail_rate=0.10, seed_name="p2:yolo"),
                lambda: SecondaryEngine("opencv_blob", skill=0.62)),
    "faces": (lambda: PrimaryEngine("facenet", skill=0.90, fail_rate=0.14, seed_name="p2:facenet"),
              lambda: SecondaryEngine("histogram", skill=0.48)),
    "anpr": (lambda: PrimaryEngine("paddleocr", skill=0.88, fail_rate=0.18, seed_name="p2:paddle"),
             lambda: SecondaryEngine("tesseract", skill=0.55)),
}


def make_samples(n: int, entity_class: str, seed_name: str) -> list[Sample]:
    g = rng(seed_name)
    out = []
    for i in range(n):
        d = float(np.clip(g.beta(2.0, 3.5), 0.0, 1.0))
        out.append(Sample(sample_id=i, difficulty=d, entity_class=entity_class, truth=f"T{i:06d}"))
    return out
