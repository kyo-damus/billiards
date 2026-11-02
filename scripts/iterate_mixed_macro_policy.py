"""Bounded collect -> MCTS visits -> Policy training iteration.

Uses the existing mixed 31-D Q, Candidate Policy, MCTS, Micro and
FastFiz without changing their semantics. Repeats a SMALL number
of iterations; each cycle gathers new independently seeded experience,
trains a fresh Policy checkpoint, then uses it in the next cycle.

This is implementation scaffolding for policy iteration, NOT
a gameplay-performance evaluation or an on-policy Q update.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from scripts.build_macro_candidate_value_dataset import (
    random_state,
    resolve_device,
)
from scripts.build_mixed_rollout_visit_dataset import (
    build_visit_dataset,
)
from scripts.collect_mixed_rollouts import collect_rollouts
from scripts.smoke_mixed_hierarchical_rollout import build_agent
from scripts.train_macro_visit_policy import train as train_visit_policy
from src.env.micro_billiard_env import MicroBilliardEnv


@dataclass(frozen=True)
class PolicyIterationRecord:
    index: int
    input_policy: str | None
    visited_roots: int
    candidate_rows: int
    output_policy: Path


def run_policy_iterations(
    args,
    *,
    agent_builder=build_agent,
    collector=collect_rollouts,
    visit_builder=build_visit_dataset,
    policy_trainer=train_visit_policy,
    state_factory=None,
):
    """Perform bounded policy updates and keep one directory per cycle.

    Check all destination paths BEFORE simulation. Never overwrite
    existing experience or checkpoints in the iteration pipeline.
    """
    if args.iterations < 1 or args.games_per_iteration < 2:
        raise ValueError("Need >=1 iteration and >=2 games per iteration.")
    if args.max_turns < 1 or args.simulations < 1:
        raise ValueError("max_turns and simulations must be positive.")
    if args.policy_epochs < 1:
        raise ValueError("policy_epochs must be positive.")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")

    root = Path(args.output_dir)
    layouts = []
    for iteration in range(args.iterations):
        folder = root / f"iteration_{iteration:03d}"
        rollout_dir = folder / "rollouts"
        visits = folder / "candidate_policy_visits.npz"
        policy = folder / "candidate_policy.pt"
        seeds = [
            args.start_seed + iteration * args.games_per_iteration + i
            for i in range(args.games_per_iteration)
        ]
        rollout_paths = [
            rollout_dir / f"rollout_seed_{seed}.npz" for seed in seeds
        ]
        layouts.append((folder, rollout_dir, visits, policy, rollout_paths))

    all_outputs = [
        path for (_, _, visits, policy, rollouts) in layouts
        for path in [visits, policy, *rollouts]
    ]
    if len({p.resolve() for p in all_outputs}) != len(all_outputs):
        raise ValueError("Duplicate iteration output paths.")
    occupied = [path for path in all_outputs if path.exists()]
    if occupied:
        raise FileExistsError(
            "Existing iteration files would be overwritten: "
            + ", ".join(str(x) for x in occupied)
        )

    if state_factory is None:
        env = MicroBilliardEnv()
        state_factory = lambda seed: random_state(env, seed)

    device = resolve_device(args.device)
    current_policy = args.candidate_policy_checkpoint
    records = []

    print("=== Mixed Macro Candidate Policy iteration smoke ===")
    print(
        f"device={device} iterations={args.iterations} "
        f"games_per_iteration={args.games_per_iteration} "
        f"max_turns={args.max_turns} simulations={args.simulations}"
    )

    for iteration, (_, rollout_dir, visits, policy, rollout_paths) in enumerate(
        layouts
    ):
        args_for_agent = SimpleNamespace(**vars(args))
        args_for_agent.candidate_policy_checkpoint = current_policy
        print(
            f"iteration={iteration}: input_policy="
            f"{current_policy if current_policy is not None else 'uniform'}"
        )
        agent = agent_builder(args_for_agent, device)
        summary = collector(
            agent,
            state_factory,
            games=args.games_per_iteration,
            start_seed=(
                args.start_seed + iteration * args.games_per_iteration
            ),
            max_turns=args.max_turns,
            output_dir=rollout_dir,
            return_dataset=None,
            gamma=args.gamma,
        )
        if summary.saved_games != args.games_per_iteration:
            raise RuntimeError("Collector returned incomplete save count.")

        _, roots, candidates, skipped = visit_builder(
            rollout_paths,
            visits,
        )
        print(
            f"iteration={iteration}: root_states={roots} "
            f"candidate_rows={candidates} skipped_roots={skipped}"
        )

        policy_trainer(
            SimpleNamespace(
                dataset=str(visits),
                checkpoint=str(policy),
                epochs=args.policy_epochs,
                hidden_dim=args.hidden_dim,
                learning_rate=args.learning_rate,
                validation_fraction=args.validation_fraction,
                seed=args.training_seed + iteration,
                device=args.device,
            )
        )
        if not policy.exists():
            raise RuntimeError("Policy trainer did not save a checkpoint.")

        records.append(
            PolicyIterationRecord(
                index=iteration,
                input_policy=(
                    None if current_policy is None else str(current_policy)
                ),
                visited_roots=roots,
                candidate_rows=candidates,
                output_policy=policy,
            )
        )
        current_policy = str(policy)
        print(f"iteration={iteration}: next_policy={current_policy}")

    print(
        "Completed bounded policy-iteration wiring. "
        "No performance improvement is implied."
    )
    return records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--candidate-value-checkpoint", required=True)
    parser.add_argument("--candidate-policy-checkpoint")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games-per-iteration", type=int, default=2)
    parser.add_argument("--start-seed", type=int, default=94100)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("macro_policy_iteration_smoke"),
    )
    parser.add_argument("--policy-epochs", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--training-seed", type=int, default=94100)
    parser.add_argument("--root-min-visits", type=int, default=1)
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
        "--safety-powers", type=float, nargs="+",
        default=[0.65, 1.10],
    )
    parser.add_argument(
        "--safety-angles", type=float, nargs="+",
        default=[-12.0, 0.0, 12.0],
    )
    parser.add_argument(
        "--safety-tip-y-mm", type=float, nargs="+", default=[0.0, 5.0]
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_policy_iterations(parse_args())
