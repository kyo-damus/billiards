import numpy as np
import torch

from scripts.train_position_residual_scale_gate import (
    choose_threshold,
    parse_scales,
    selection_metrics,
)


def test_parse_scales():

    scales = parse_scales(
        "0,0.25,0.5,0.75,1.0"
    )

    assert scales == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]


def test_selection_metrics_uses_successful_scale():

    logits = torch.tensor(
        [
            [0.0, 0.1, 3.0],
            [0.0, 3.0, 0.1],
        ],
        dtype=torch.float32,
    )

    success = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    (
        overall,
        capable_rate,
        selected,
    ) = selection_metrics(
        logits,
        success,
        zero_index=0,
        threshold=0.0,
    )

    assert np.isclose(
        overall,
        1.0,
    )
    assert np.isclose(
        capable_rate,
        1.0,
    )

    np.testing.assert_array_equal(
        selected.cpu().numpy(),
        np.array(
            [2, 1],
            dtype=np.int64,
        ),
    )


def test_choose_threshold_returns_valid_value():

    logits = torch.zeros(
        (4, 3),
        dtype=torch.float32,
    )

    success = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    threshold, _, _ = choose_threshold(
        logits,
        success,
        zero_index=0,
    )

    assert (
        0.0
        <= threshold
        <= 0.95
    )
