import torch

from scripts.train_position_probability_gate import (
    choose_threshold,
    selection_metrics,
)


def test_selection_metrics_falls_back_to_zero():

    logits = torch.tensor(
        [
            [-3.0, -2.0, -1.0],
        ]
    )

    success = torch.tensor(
        [
            [1.0, 0.0, 0.0],
        ]
    )

    (
        overall,
        capable,
        selected,
    ) = selection_metrics(
        logits,
        success,
        zero_index=0,
        threshold=0.50,
    )

    assert int(
        selected.item()
    ) == 0

    assert overall == 1.0
    assert capable == 1.0


def test_choose_threshold_returns_valid_value():

    logits = torch.tensor(
        [
            [3.0, 0.0],
            [0.0, 3.0],
        ]
    )

    success = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    (
        threshold,
        overall,
        capable,
    ) = choose_threshold(
        logits,
        success,
        zero_index=0,
    )

    assert (
        0.0 <= threshold <= 0.95
    )
    assert overall == 1.0
    assert capable == 1.0
