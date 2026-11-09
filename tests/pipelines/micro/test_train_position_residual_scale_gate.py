import numpy as np
import torch

from scripts.train_position_residual_scale_gate import (
    build_gate_loss,
    choose_threshold,
    multi_positive_set_loss,
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



def test_multi_positive_set_loss_accepts_any_successful_scale():

    logits_good = torch.tensor(
        [
            [0.0, 2.0, 1.5],
        ],
        dtype=torch.float32,
    )

    logits_bad = torch.tensor(
        [
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    success = torch.tensor(
        [
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    good_loss = multi_positive_set_loss(
        logits_good,
        success,
    )
    bad_loss = multi_positive_set_loss(
        logits_bad,
        success,
    )

    assert (
        float(good_loss.item())
        < float(bad_loss.item())
    )


def test_success_set_loss_falls_back_to_base_on_incapable_rows():

    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    success = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    pos_weight = torch.ones(
        3,
        dtype=torch.float32,
    )

    loss = build_gate_loss(
        logits=logits,
        success_masks=success,
        zero_index=0,
        objective="success_set",
        pos_weight=pos_weight,
        capable_weight=2.0,
        negative_base_weight=0.25,
    )

    assert torch.isfinite(
        loss
    )

    loss.backward()

    assert logits.grad is not None
