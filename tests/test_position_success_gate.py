import torch

from scripts.train_position_success_gate import (
    multi_positive_set_loss,
)


def test_multi_positive_set_loss_prefers_any_success():

    success = torch.tensor(
        [
            [0.0, 1.0, 1.0],
        ]
    )

    good_logits = torch.tensor(
        [
            [0.0, 3.0, 0.0],
        ]
    )

    bad_logits = torch.tensor(
        [
            [3.0, 0.0, 0.0],
        ]
    )

    good_loss = (
        multi_positive_set_loss(
            good_logits,
            success,
        )
    )

    bad_loss = (
        multi_positive_set_loss(
            bad_logits,
            success,
        )
    )

    assert (
        good_loss.item()
        < bad_loss.item()
    )


def test_multi_positive_set_loss_all_negative_safe():

    logits = torch.zeros(
        2,
        3,
        requires_grad=True,
    )

    success = torch.zeros(
        2,
        3,
    )

    loss = multi_positive_set_loss(
        logits,
        success,
    )

    loss.backward()

    assert loss.item() == 0.0
