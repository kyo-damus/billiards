"""A reproducible partition of the 7! orders of the unfixed balls in nine-ball."""

from dataclasses import dataclass
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path

import numpy as np


BALLS = tuple(range(2, 9))
ALL_ORDERS = tuple(permutations(BALLS))


def legacy_rack_order(seed):
    return tuple(int(ball) for ball in np.random.default_rng(seed).permutation(BALLS))


@dataclass(frozen=True)
class RackLayoutSplit:
    training: tuple
    evaluation: tuple
    sha256: str

    def order(self, seed, *, evaluation=False):
        pool = self.evaluation if evaluation else self.training
        digest = sha256(str(seed).encode("ascii")).digest()
        return pool[int.from_bytes(digest, "big") % len(pool)]


def load_rack_layout_split(path):
    contents = Path(path).read_bytes()
    payload = json.loads(contents)
    if payload.get("kind") != "nine_ball_rack_order_split_v1":
        raise ValueError(f"Unsupported rack layout split: {path}")
    evaluation = tuple(tuple(order) for order in payload["evaluation_orders"])
    heldout = set(evaluation)
    universe = set(ALL_ORDERS)
    if (
        not 0 < len(heldout) < len(universe)
        or len(evaluation) != len(heldout)
        or not heldout <= universe
    ):
        raise ValueError(f"Invalid evaluation orders in rack layout split: {path}")
    training = tuple(order for order in ALL_ORDERS if order not in heldout)
    return RackLayoutSplit(training, evaluation, sha256(contents).hexdigest())
