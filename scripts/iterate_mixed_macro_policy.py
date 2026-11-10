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
from scripts.build_mixed_rollout_return_dataset import (
    build_return_dataset,
)
from scripts.train_macro_candidate_value import (
    train as train_candidate_q,
)
from scripts.train_macro_state_value import (
    train as train_state_value,
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
    output_q: Path | None = None
    output_v: Path | None = None
    complete_games: int = 0


def run_policy_iterations(
    args,
    *,
    agent_builder=build_agent,
    collector=collect_rollouts,
    visit_builder=build_visit_dataset,
    policy_trainer=train_visit_policy,
    state_factory=None,
    return_builder=build_return_dataset,
    q_trainer=train_candidate_q,
    value_trainer=train_state_value,
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
    update_critics = bool(getattr(args, "update_critics", False))
    critic_epochs = int(getattr(args, "critic_epochs", 2))
    if update_critics and critic_epochs < 1:
        raise ValueError("critic_epochs must be positive.")

    root = Path(args.output_dir)
    layouts = []
    for iteration in range(args.iterations):
        folder = root / f"iteration_{iteration:03d}"
        rollout_dir = folder / "rollouts"
        visits = folder / "candidate_policy_visits.npz"
        policy = folder / "candidate_policy.pt"
        returns = folder / "completed_action_returns.npz"
        q_path = folder / "candidate_q.pt"
        v_path = folder / "state_value.pt"
        seeds = [
            args.start_seed + iteration * args.games_per_iteration + i
            for i in range(args.games_per_iteration)
        ]
        rollout_paths = [
            rollout_dir / f"rollout_seed_{seed}.npz" for seed in seeds
        ]
        layouts.append((
            folder, rollout_dir, visits, policy,
            returns, q_path, v_path, rollout_paths,
        ))

    all_outputs = [
        path
        for (
            _, _, visits, policy, returns, q_path, v_path, rollouts
        ) in layouts
        for path in (
            [visits, policy, *rollouts]
            + ([returns, q_path, v_path] if update_critics else [])
        )
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
    current_q = getattr(args, "candidate_value_checkpoint", None)
    current_v = getattr(args, "state_value_checkpoint", None)
    records = []

    print("=== Mixed Macro Candidate Policy iteration smoke ===")
    print(
        f"device={device} iterations={args.iterations} "
        f"games_per_iteration={args.games_per_iteration} "
        f"max_turns={args.max_turns} simulations={args.simulations}"
    )

    for iteration, (
        _, rollout_dir, visits, policy,
        returns, q_path, v_path, rollout_paths,
    ) in enumerate(layouts):
        args_for_agent = SimpleNamespace(**vars(args))
        args_for_agent.candidate_policy_checkpoint = current_policy
        args_for_agent.candidate_value_checkpoint = current_q
        args_for_agent.state_value_checkpoint = current_v
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

        critic_output_q = None
        critic_output_v = None
        completed_paths = tuple(getattr(summary, "completed_paths", ()))
        if update_critics:
            if len(completed_paths) < 2:
                print(
                    f"iteration={iteration}: completed_games="
                    f"{len(completed_paths)} < 2; skip Q/V updates. "
                    "Keep previous Q and V checkpoints."
                )
            else:
                return_builder(
                    completed_paths,
                    returns,
                    gamma=args.gamma,
                )
                common = dict(
                    dataset=str(returns),
                    epochs=critic_epochs,
                    hidden_dim=args.hidden_dim,
                    learning_rate=args.learning_rate,
                    validation_fraction=args.validation_fraction,
                    seed=args.training_seed + iteration,
                    device=args.device,
                )
                q_trainer(SimpleNamespace(
                    **common,
                    checkpoint=str(q_path),
                    batch_size=64,
                    weight_decay=0.0,
                    grad_clip=5.0,
                    log_interval=1,
                ))
                value_trainer(SimpleNamespace(
                    **common,
                    checkpoint=str(v_path),
                    batch_size=64,
                ))
                if not q_path.exists() or not v_path.exists():
                    raise RuntimeError(
                        "Q/V trainers failed to save both checkpoints."
                    )
                critic_output_q = q_path
                critic_output_v = v_path
                current_q = str(q_path)
                current_v = str(v_path)
                print(
                    f"iteration={iteration}: updated Q={q_path} "
                    f"V={v_path}"
                )

        records.append(
            PolicyIterationRecord(
                index=iteration,
                input_policy=(
                    None if current_policy is None else str(current_policy)
                ),
                visited_roots=roots,
                candidate_rows=candidates,
                output_policy=policy,
                output_q=critic_output_q,
                output_v=critic_output_v,
                complete_games=len(completed_paths),
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
    parser.add_argument(
        "--state-value-checkpoint",
        help="Optional trained 12D state V; otherwise zero leaf value.",
    )
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
    parser.add_argument(
        "--update-critics", action="store_true",
        help=(
            "Optional: update Candidate Q and State V on complete "
            "game returns only, when >=2 games finish in a batch."
        ),
    )
    parser.add_argument("--critic-epochs", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--training-seed", type=int, default=94100)
    parser.add_argument("--root-min-visits", type=int, default=1)
    parser.add_argument(
        "--root-tie-breaker",
        choices=("first", "empirical_q"),
        default="first",
        help="Opt-in tie handling; default preserves original search.",
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
    args = parser.parse_args()
    if args.critic_epochs < 1:
        parser.error("--critic-epochs must be positive.")
    return args


if __name__ == "__main__":
    run_policy_iterations(parse_args())
