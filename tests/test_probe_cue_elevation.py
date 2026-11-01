"""Standalone cue-elevation diagnostic: counters do not alter rewards."""

from types import SimpleNamespace

import numpy as np

from scripts.probe_cue_elevation_on_hard_states import (
    attack_precondition_summary,
    rail_distances,
    summarize_code_counts,
)
from src.common.table import TABLE_LENGTH, TABLE_WIDTH
from src.common.types import GameState


def make_state():
    return GameState(
        ball_positions=np.asarray(
            [[0.5, 0.04], [0.5, 0.8], [0.9, 1.5]],
            dtype=np.float32,
        ),
        score=np.zeros(2, dtype=np.float32),
        current_player=0,
    )


def test_nearest_rail_distances_use_table_geometry():
    distances = rail_distances(make_state().ball_positions)
    assert np.isclose(distances["west"], 0.5)
    assert np.isclose(distances["east"], TABLE_WIDTH - 0.5)
    assert np.isclose(distances["south"], 0.04)
    assert np.isclose(distances["north"], TABLE_LENGTH - 0.04)
    assert min(distances, key=distances.get) == "south"


def test_rejection_bitmasks_are_descriptive():
    labels = summarize_code_counts({128: 24, 8: 2})
    assert "128:CUE_STICK_COLLISION=24" in labels
    assert "8:BAD_PHI_VAL=2" in labels


def test_attack_physics_rejections_not_counted_as_real_misses():
    class Attack:
        def __init__(self):
            self.last_shot_info = None

        def step(self, state, candidate):
            if candidate == 0:
                self.last_shot_info = {
                    "physically_possible": False,
                    "precondition_code": 128,
                }
                target_potted = False
            elif candidate == 1:
                self.last_shot_info = {
                    "physically_possible": True,
                    "precondition_code": 0,
                }
                target_potted = False
            else:
                self.last_shot_info = {
                    "physically_possible": True,
                    "precondition_code": 0,
                }
                target_potted = True
            return SimpleNamespace(
                evaluation=SimpleNamespace(
                    target_potted=target_potted,
                    scratch=False,
                )
            )

    attack = Attack()
    mixed = SimpleNamespace(attack_transition_model=attack)
    result = attack_precondition_summary(
        make_state(), [0, 1, 2], mixed
    )

    assert result["physics_invalid"] == 1
    assert result["precondition_128"] == 1
    assert result["valid_no_pot"] == 1
    assert result["valid_pot"] == 1
