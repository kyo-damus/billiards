"""Collect bounded mixed hierarchical games with per-game experience logs.

Implementation smoke: reuses existing learned 31-D Policy/Q and MCTS.
All rollouts are saved, including truncated/no-action endings.
Only COMPLETE games can be converted to observed-return Q labels.
This script does not train or estimate performance.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.build_macro_candidate_value_dataset import (
    random_state,
    resolve_device,
)
from scripts.build_mixed_rollout_return_dataset import (
    build_return_dataset,
)
from scripts.smoke_mixed_hierarchical_rollout import build_agent
from src.controller.mixed_hierarchical_rollout import (
    MixedHierarchicalRollout,
)
from src.controller.mixed_rollout_dataset import (
    save_mixed_rollout_npz,
)
from src.env.micro_billiard_env import MicroBilliardEnv


@dataclass(frozen=True)
class CollectionSummary:
    saved_games: int
    completed_games: int
    unfinished_games: int
    total_executed_turns: int
    completed_executed_turns: int
    completed_paths: tuple[Path, ...]
    return_dataset: Path | None


def collect_rollouts(
    agent,
    state_factory,
    *,
    games,
    start_seed,
    max_turns,
    output_dir,
    return_dataset=None,
    gamma=0.99,
    overwrite=False,
):
    """Collect independently seeded games, never label a partial game.

    Paths are checked for conflicts *before* starting simulation.
    The same agent/MCTS may be reused, but each rollout starts
    from an independently generated GameState.
    """
    if games < 1 or max_turns < 1:
        raise ValueError("games and max_turns must be positive.")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")

    output_dir = Path(output_dir)
    paths = [
        output_dir / f"rollout_seed_{start_seed + index}.npz"
        for index in range(games)
    ]
    dataset_path = (
        Path(return_dataset) if return_dataset is not None else None
    )

    all_outputs = paths + (
        [dataset_path] if dataset_path is not None else []
    )
    if len({p.resolve() for p in all_outputs}) != len(all_outputs):
        raise ValueError("Output paths must be distinct.")
    if not overwrite:
        existing = [p for p in all_outputs if p.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing collection output: "
                + ", ".join(str(path) for path in existing)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    runner = MixedHierarchicalRollout(agent)
    completed_paths = []
    total_turns = 0
    completed_turns = 0

    for seed, path in zip(range(start_seed, start_seed + games), paths):
        initial = state_factory(seed)
        rollout = runner.run(initial, max_turns=max_turns)
        save_mixed_rollout_npz(rollout, path, seed=seed)

        steps = len(rollout.records)
        total_turns += steps
        completed = bool(
            rollout.terminated
            and rollout.stop_reason == "game_terminated"
            and steps > 0
        )
        if completed:
            completed_paths.append(path)
            completed_turns += steps

        print(
            f"seed={seed} turns={steps} "
            f"stop={rollout.stop_reason} "
            f"completed={completed} "
            f"score={rollout.final_state.score.tolist()} "
            f"saved={path}"
        )

    created_dataset = None
    if dataset_path is not None:
        if completed_paths:
            result_path, episode_count, sample_count = (
                build_return_dataset(
                    completed_paths,
                    dataset_path,
                    gamma=gamma,
                )
            )
            created_dataset = result_path
            print(
                "completed-only selected-action return dataset: "
                f"{result_path} (games={episode_count}, "
                f"executed actions={sample_count})"
            )
        else:
            print(
                "No completed game: return dataset not created. "
                "Truncated experience files were retained."
            )

    summary = CollectionSummary(
        saved_games=games,
        completed_games=len(completed_paths),
        unfinished_games=games - len(completed_paths),
        total_executed_turns=total_turns,
        completed_executed_turns=completed_turns,
        completed_paths=tuple(completed_paths),
        return_dataset=created_dataset,
    )
    print(
        "collection summary: "
        f"saved={summary.saved_games} "
        f"completed={summary.completed_games} "
        f"unfinished={summary.unfinished_games} "
        f"executed_turns={summary.total_executed_turns} "
        f"completed_turns={summary.completed_executed_turns}"
    )
    return summary


def main(args):
    device = resolve_device(args.device)
    agent = build_agent(args, device)
    env = MicroBilliardEnv()

    print("=== Bounded mixed hierarchical game collection ===")
    print(
        f"device={device} games={args.games} "
        f"start_seed={args.start_seed} "
        f"max_turns={args.max_turns} "
        f"MCTS_simulations={args.simulations}"
    )
    print(
        "This collects implementation experience, NOT an unbiased "
        "performance benchmark or new trained policy."
    )
    collect_rollouts(
        agent,
        lambda seed: random_state(env, seed),
        games=args.games,
        start_seed=args.start_seed,
        max_turns=args.max_turns,
        output_dir=args.output_dir,
        return_dataset=args.return_dataset,
        gamma=args.gamma,
        overwrite=args.overwrite,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--candidate-value-checkpoint", required=True)
    parser.add_argument("--candidate-policy-checkpoint")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--start-seed", type=int, default=94000)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("mixed_rollouts")
    )
    parser.add_argument(
        "--return-dataset", type=Path, default=None,
        help=(
            "If provided, combine ONLY completed games into an "
            "executed-action return dataset. Otherwise collect only."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Explicitly allow replacing existing rollouts/dataset.",
    )
    parser.add_argument("--root-min-visits", type=int, default=1)
    parser.add_argument(
        "--root-tie-breaker",
        choices=("first", "empirical_q"),
        default="first",
        help="Root equal-visit tie behavior; default preserves old runs.",
    )
    parser.add_argument(
        "--candidate-q-prior-visits", type=float, default=2.0
    )
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--residual-scale", type=float, default=0.75)
    parser.add_argument("--radius", type=float, default=0.10)
    parser.add_argument("--outcome-grid-size", type=int, default=5)
    parser.add_argument("--outcome-top-k", type=int, default=8)
    parser.add_argument(
        "--outcome-dedup-distance", type=float, default=0.10
    )
    parser.add_argument("--safety-top-k", type=int, default=4)
    parser.add_argument(
        "--safety-powers", type=float, nargs="+", default=[0.65, 1.10]
    )
    parser.add_argument(
        "--safety-angles", type=float, nargs="+",
        default=[-12.0, 0.0, 12.0],
    )
    parser.add_argument(
        "--safety-tip-y-mm", type=float, nargs="+", default=[0.0, 5.0]
    )
    args = parser.parse_args()
    if args.games < 1 or args.max_turns < 1 or args.simulations < 1:
        parser.error(
            "--games, --max-turns and --simulations must be positive."
        )
    if args.root_min_visits < 0 or args.candidate_q_prior_visits < 0:
        parser.error("Root min visits / Q prior visits must be >= 0.")
    if not np.isfinite(args.gamma) or not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must be in [0, 1].")
    return args


if __name__ == "__main__":
    main(parse_args())
