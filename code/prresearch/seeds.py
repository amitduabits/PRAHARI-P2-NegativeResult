"""Single source of randomness for every experiment in this package.

Every experiment script calls `rng(name)` instead of touching numpy.random or
random directly. Two runs of the same script on the same machine and the same
package version produce byte-identical results files.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

MASTER_SEED = 20260905


def derive(name: str) -> int:
    digest = hashlib.sha256(f"{MASTER_SEED}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def rng(name: str) -> np.random.Generator:
    return np.random.default_rng(derive(name))


def pyrandom(name: str) -> random.Random:
    return random.Random(derive(name))
