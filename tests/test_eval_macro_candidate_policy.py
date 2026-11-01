import numpy as np

from scripts.eval_macro_candidate_policy import (
    calculate_state_metrics,
)


def test_state_metrics_accept_tied_best_q():

    probability = np.array(
        [
            0.10,
            0.60,
            0.30,
        ],
        dtype=np.float32,
    )

    target_q = np.array(
        [
            2.0,
            2.0,
            1.0,
        ],
        dtype=np.float32,
    )

    metrics = calculate_state_metrics(
        probability,
        target_q,
    )

    assert metrics[
        "top1_correct"
    ] == 1.0
    assert metrics[
        "top3_hit"
    ] == 1.0
    assert np.isclose(
        metrics[
            "regret"
        ],
        0.0,
    )
    assert np.isclose(
        metrics[
            "optimal_prior_mass"
        ],
        0.70,
    )


def test_state_metrics_reports_regret():

    probability = np.array(
        [
            0.60,
            0.30,
            0.10,
        ],
        dtype=np.float32,
    )

    target_q = np.array(
        [
            1.0,
            2.0,
            0.0,
        ],
        dtype=np.float32,
    )

    metrics = calculate_state_metrics(
        probability,
        target_q,
    )

    assert metrics[
        "top1_correct"
    ] == 0.0
    assert np.isclose(
        metrics[
            "regret"
        ],
        1.0,
    )
