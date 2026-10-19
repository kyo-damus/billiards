import numpy as np

from scripts.eval_position_residual_sweep import (
    aggregate,
)


def test_aggregate_mean_and_std():

    rows = [
        {
            "pot": 0.8,
            "cue_region": 0.1,
            "goal": 0.1,
            "scratch": 0.05,
            "wrong_pocket": 0.0,
            "mean_cue_distance": 0.5,
            "mean_goal_reward": 10.0,
            "mean_residual_norm": 0.3,
            "clip_rate": 0.0,
        },
        {
            "pot": 0.9,
            "cue_region": 0.2,
            "goal": 0.2,
            "scratch": 0.03,
            "wrong_pocket": 0.01,
            "mean_cue_distance": 0.4,
            "mean_goal_reward": 11.0,
            "mean_residual_norm": 0.5,
            "clip_rate": 0.1,
        },
    ]

    metrics = aggregate(
        rows
    )

    assert np.isclose(
        metrics["pot"]["mean"],
        0.85,
    )
    assert np.isclose(
        metrics[
            "mean_cue_distance"
        ][
            "mean"
        ],
        0.45,
    )
    assert metrics[
        "clip_rate"
    ][
        "std"
    ] > 0.0
