"""Standalone cue-elevation diagnostic: counters do not alter rewards."""

from types import SimpleNamespace

import numpy as np

from scripts.probe_cue_elevation_on_hard_states import (
    attack_precondition_summary,
    rail_distances,
    summarize_code_counts,
    probe_elevation_grid,
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



def test_probe_power_tip_grid_tracks_first_contact(monkeypatch):
    from types import SimpleNamespace
    import scripts.probe_cue_elevation_on_hard_states as module

    observed = []

    class FakeSimulator:
        def restore(self, snapshot):
            self.current = snapshot

        def step(self, target_ball, power, angle, tip_x_mm=0.0,
                 tip_y_mm=0.0, cue_elevation_deg=10.0):
            observed.append((power, tip_y_mm, cue_elevation_deg))
            return SimpleNamespace(
                precondition_code=0,
                placement_code=0,
                first_contact_ball=(target_ball if power > 1.0 else -1),
            )

        def snapshot(self):
            return self.current

    monkeypatch.setattr(module, "FastFizSimulator", FakeSimulator)
    rows = probe_elevation_grid(
        make_state(),
        elevations=[20.0, 35.0],
        angle_offsets=[0.0],
        powers=[0.8, 1.2],
        tip_y_mm=[0.0, 5.0],
    )

    assert len(rows) == 2
    for row in rows:
        assert row["attempted"] == 8
        assert row["valid"] == 8
        assert row["any_first"] == 4
        assert row["correct_first"] == 4
        assert row["safety_like"] == 4
        assert row["first_contact_example"]["power"] == 1.2
        assert row["first_contact_example"]["tip_y_mm"] == 0.0

    assert {v[0] for v in observed} == {0.8, 1.2}
    assert {v[1] for v in observed} == {0.0, 5.0}
    assert {v[2] for v in observed} == {20.0, 35.0}


def test_probe_rejects_out_of_range_power_and_tip():
    import pytest

    with pytest.raises(ValueError, match="power/tip"):
        probe_elevation_grid(
            make_state(),
            elevations=[20.0],
            angle_offsets=[0.0],
            powers=[-1.0],
        )
    with pytest.raises(ValueError, match="power/tip"):
        probe_elevation_grid(
            make_state(),
            elevations=[20.0],
            angle_offsets=[0.0],
            powers=[1.0],
            tip_y_mm=[13.0],
        )

def test_probe_turn_and_threat_sample_limit(monkeypatch):
    from types import SimpleNamespace
    import scripts.probe_cue_elevation_on_hard_states as module

    class Sim:
        def restore(self, snapshot):
            self.snapshot_value = snapshot

        def step(self, target_ball, **kwargs):
            return SimpleNamespace(
                precondition_code=0,
                placement_code=0,
                first_contact_ball=target_ball,
            )

        def snapshot(self):
            return self.snapshot_value

    class Threat:
        def __init__(self):
            self.count = 0

        def evaluate(self, state, defending_player):
            assert state.current_player != defending_player
            self.count += 1
            return SimpleNamespace(successful_pot_count=1)

    monkeypatch.setattr(module, "FastFizSimulator", Sim)
    t = Threat()
    rows = probe_elevation_grid(
        make_state(),
        elevations=[20.0],
        angle_offsets=[0.0],
        powers=[1.0],
        tip_y_mm=[0.0, 5.0],
        threat_evaluator=t,
        threat_samples_per_elevation=1,
    )
    assert rows[0]["eligible_safety"] == 4
    assert rows[0]["sampled_threat_count"] == 1
    assert rows[0]["min_opponent_direct_success"] == 1
    assert t.count == 1



def test_threat_sampling_can_compare_trained_and_physical_opponent(
    monkeypatch,
):
    import scripts.probe_cue_elevation_on_hard_states as module

    class FakeSimulator:
        def restore(self, snapshot):
            self.current = snapshot

        def step(self, target_ball, **kwargs):
            return SimpleNamespace(
                precondition_code=0,
                placement_code=0,
                first_contact_ball=target_ball,
            )

        def snapshot(self):
            return self.current

    class TrainedThreat:
        def evaluate(self, state, defending_player):
            return SimpleNamespace(successful_pot_count=1)

    class PhysicalThreat:
        def __init__(self):
            self.calls = 0

        def evaluate(self, state):
            self.calls += 1
            return SimpleNamespace(
                successful_routes=2,
                attempted=12,
                physically_valid=8,
            )

    monkeypatch.setattr(module, "FastFizSimulator", FakeSimulator)
    physical = PhysicalThreat()
    rows = probe_elevation_grid(
        make_state(),
        elevations=[20.0],
        angle_offsets=[0.0],
        powers=[1.0],
        tip_y_mm=[0.0, 5.0],
        threat_evaluator=TrainedThreat(),
        elevated_direct_probe=physical,
        threat_samples_per_elevation=1,
    )
    row = rows[0]
    assert row["eligible_safety"] == 4
    assert row["sampled_threat_count"] == 1
    assert row["min_opponent_direct_success"] == 1
    assert row["min_opponent_elevated_direct_success"] == 2
    assert row["max_opponent_elevated_direct_success"] == 2
    assert row["elevated_direct_attempts"] == 12
    assert row["elevated_direct_physics_valid"] == 8
    assert physical.calls == 1
