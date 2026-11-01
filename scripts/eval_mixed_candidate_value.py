"""Evaluate a 31-D mixed Attack/Safety Candidate Q on held-out boards.

Input is a separately generated dataset. All metrics are relative
to deterministic two-step FastFiz teachers, NOT actual win rate.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.build_macro_candidate_value_dataset import resolve_device
from scripts.train_macro_candidate_value import (
    calculate_metrics,
    load_dataset,
)
from src.macro.candidate_value import (
    load_macro_candidate_value_network,
)


def inspect_mixed_metadata(path):
    """Return reproducibility metadata and validate mixed dataset."""
    with np.load(path) as data:
        if "candidate_encoding_version" not in data:
            raise ValueError("Missing candidate_encoding_version.")
        if int(data["candidate_encoding_version"]) != 2:
            raise ValueError("Expected candidate encoding version 2.")
        if "turn_aware" not in data or not bool(
            data["turn_aware"]
        ):
            raise ValueError("Mixed evaluation requires turn_aware=True.")
        if "strategy" not in data:
            raise ValueError("Mixed evaluation requires strategy labels.")

        strategy = np.asarray(
            data["strategy"], dtype=np.int64
        ).reshape(-1)

        if not np.all(np.isin(strategy, (0, 1))):
            raise ValueError("Strategies must be 0=Attack or 1=Safety.")

        start = (
            int(data["seed_start"])
            if "seed_start" in data
            else None
        )
        count = (
            int(data["requested_states"])
            if "requested_states" in data
            else None
        )

        return strategy, start, count


def validate_heldout_seeds(
    training_seed,
    training_states,
    evaluation_seed,
    evaluation_states,
):
    """Reject overlapping board-seed intervals when recorded."""
    if (
        training_seed is None
        or training_states is None
        or evaluation_seed is None
        or evaluation_states is None
    ):
        return False

    overlap = (
        training_seed < evaluation_seed + evaluation_states
        and evaluation_seed < training_seed + training_states
    )
    if overlap:
        raise ValueError(
            "Training and evaluation board seed ranges overlap."
        )
    return True


def ranking_summary(prediction, target, state_ids, strategies):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    state_ids = np.asarray(state_ids, dtype=np.int64)
    strategies = np.asarray(strategies, dtype=np.int64)

    lengths = {
        len(prediction), len(target),
        len(state_ids), len(strategies),
    }
    if len(lengths) != 1 or not len(prediction):
        raise ValueError("Nonempty equal-length vectors are required.")

    summary = {
        "boards": 0,
        "boards_with_safety": 0,
        "selected_safety": 0,
        "oracle_safety_competitive": 0,
        "top1_hits": 0,
        "mean_regret": 0.0,
    }
    regrets = []

    for board_id in np.unique(state_ids):
        indices = np.flatnonzero(state_ids == board_id)
        if not len(indices):
            continue

        summary["boards"] += 1
        selected = indices[
            int(np.argmax(prediction[indices]))
        ]
        best_target = float(np.max(target[indices]))
        selected_target = float(target[selected])

        summary["selected_safety"] += int(
            strategies[selected] == 1
        )
        summary["top1_hits"] += int(
            np.isclose(
                selected_target, best_target,
                atol=1e-6,
            )
        )
        regrets.append(max(0.0, best_target - selected_target))

        safety = indices[strategies[indices] == 1]
        if len(safety):
            summary["boards_with_safety"] += 1
            summary["oracle_safety_competitive"] += int(
                np.isclose(
                    np.max(target[safety]),
                    best_target,
                    atol=1e-6,
                )
            )

    summary["mean_regret"] = float(np.mean(regrets))
    return summary


def main(args):
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    eval_path = Path(args.dataset).resolve()

    if not eval_path.exists():
        raise FileNotFoundError(eval_path)

    if checkpoint_path == eval_path:
        raise ValueError("Checkpoint and dataset must be separate.")

    model, checkpoint = load_macro_candidate_value_network(
        checkpoint_path,
        device=device,
        expected_candidate_dim=31,
    )
    if int(checkpoint.get("candidate_encoding_version", -1)) != 2:
        raise ValueError("Checkpoint is not mixed Candidate Q v2.")
    if not bool(checkpoint.get("turn_aware", False)):
        raise ValueError("Checkpoint is not turn-aware.")

    training_path = Path(checkpoint["dataset"]).resolve()
    if eval_path == training_path:
        raise ValueError(
            "Evaluation dataset is the training dataset."
        )

    states, candidates, targets, state_ids = load_dataset(eval_path)
    strategies, eval_seed, eval_count = inspect_mixed_metadata(eval_path)

    if strategies.shape != targets.shape:
        raise ValueError("Strategy sample count does not match labels.")

    train_seed = train_count = None
    if training_path.exists():
        _, train_seed, train_count = inspect_mixed_metadata(
            training_path
        )

    verified = validate_heldout_seeds(
        train_seed, train_count, eval_seed, eval_count
    )

    state_tensor = torch.as_tensor(
        states, device=device, dtype=torch.float32
    )
    candidate_tensor = torch.as_tensor(
        candidates, device=device, dtype=torch.float32
    )

    with torch.no_grad():
        prediction = model(
            state_tensor,
            candidate_tensor,
        ).detach().cpu().numpy().reshape(-1)

    all_metrics = calculate_metrics(
        prediction, targets, state_ids
    )
    ranks = ranking_summary(
        prediction, targets, state_ids, strategies
    )

    print("=== Held-out mixed Candidate Q v2 ===")
    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"evaluation dataset: {eval_path}")
    print(
        "Disjoint seed ranges verified: "
        f"{verified} (metadata may be missing)"
    )
    print(
        "boards / samples: "
        f"{ranks['boards']} / {len(targets)}"
    )
    print(
        "MSE / MAE / top1 accuracy: "
        f"{all_metrics['mse']:.4f} / "
        f"{all_metrics['mae']:.4f} / "
        f"{all_metrics['top1_accuracy']:.3f}"
    )
    print(
        f"mean two-step ranking regret: "
        f"{ranks['mean_regret']:.4f}"
    )
    print(
        "predicted Safety selections: "
        f"{ranks['selected_safety']}/{ranks['boards']}"
    )
    print(
        "Safety matches oracle best: "
        f"{ranks['oracle_safety_competitive']}/"
        f"{ranks['boards_with_safety']} "
        "(among boards having Safety candidates)"
    )

    for strategy_id, name in ((0, "Attack"), (1, "Safety")):
        mask = strategies == strategy_id
        if not np.any(mask):
            print(f"{name}: no samples")
            continue
        error = prediction[mask] - targets[mask]
        print(
            f"{name}: n={int(mask.sum())} "
            f"target mean={float(np.mean(targets[mask])):.3f} "
            f"pred mean={float(np.mean(prediction[mask])):.3f} "
            f"MSE={float(np.mean(error ** 2)):.4f}"
        )

    print(
        "These metrics measure agreement with selected two-step "
        "teacher targets, not independent game outcome or "
        "generalization of an unlearned Safety Micro policy."
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
