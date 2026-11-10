"""Train all mixed Macro components, then evaluate on disjoint seeded games.

Phase 6 integrated execution, not a win-rate benchmark:
Macro 31-D Policy + Q + 12-D V + MCTS -> goal-conditioned Micro
-> FastFiz -> simplified two-object-ball game.

All outputs go under a NEW directory; old checkpoints are never replaced.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.build_macro_candidate_value_dataset import random_state, resolve_device
from scripts.collect_mixed_rollouts import collect_rollouts
from scripts.iterate_mixed_macro_policy import run_policy_iterations
from scripts.smoke_mixed_hierarchical_rollout import build_agent
from src.pipelines.rollout_analysis import summarize_rollouts


def _overlap(first_start, first_count, second_start, second_count):
    return (
        first_start < second_start + second_count
        and second_start < first_start + first_count
    )


def _validate(args):
    if args.iterations < 1 or args.train_games_per_iteration < 2:
        raise ValueError("Need positive iterations and >=2 training games per iteration.")
    if args.eval_games < 1 or args.max_turns < 1:
        raise ValueError("Need positive eval_games and max_turns.")
    if args.train_simulations < 1 or args.eval_simulations < 1:
        raise ValueError("MCTS simulations must be positive.")
    if args.policy_epochs < 1 or args.critic_epochs < 1:
        raise ValueError("Training epochs must be positive.")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0,1).")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("gamma must be in [0,1].")
    if args.root_min_visits < 0 or args.candidate_q_prior_visits < 0:
        raise ValueError("Root visit and Q-prior settings must be nonnegative.")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse experiment directory: {args.output_dir}")
    if _overlap(
        args.start_seed,
        args.iterations * args.train_games_per_iteration,
        args.eval_start_seed,
        args.eval_games,
    ):
        raise ValueError("Training and evaluation seed ranges overlap.")
    for path in (
        args.base_checkpoint,
        args.residual_checkpoint,
        args.candidate_policy_checkpoint,
        args.candidate_value_checkpoint,
        args.state_value_checkpoint,
    ):
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"Missing input checkpoint: {path}")


def final_checkpoints(records, initial_q, initial_v):
    """Take latest successfully updated Q/V, even if last cycle skips critics."""
    if not records:
        raise ValueError("No training iterations completed.")
    policy = Path(records[-1].output_policy)
    if not policy.is_file():
        raise FileNotFoundError(f"Missing final policy checkpoint: {policy}")
    q, v = str(initial_q), str(initial_v) if initial_v is not None else None
    for record in records:
        if (record.output_q is None) != (record.output_v is None):
            raise ValueError("Candidate Q and state V must update together.")
        if record.output_q is not None:
            if not Path(record.output_q).is_file() or not Path(record.output_v).is_file():
                raise FileNotFoundError("Missing updated Q/V checkpoint.")
            q, v = str(record.output_q), str(record.output_v)
    return str(policy), q, v


def run_experiment(
    args,
    *,
    iterate=run_policy_iterations,
    collect=collect_rollouts,
    agent_factory=build_agent,
    state_factory=None,
    report= summarize_rollouts,
):
    """Create one full train -> held-out eval result with baseline and learned."""
    _validate(args)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    if state_factory is None:
        from src.env.micro_billiard_env import MicroBilliardEnv

        env = MicroBilliardEnv()
        state_factory = lambda seed: random_state(env, seed)

    training_args = SimpleNamespace(**vars(args))
    training_args.games_per_iteration = args.train_games_per_iteration
    training_args.simulations = args.train_simulations
    training_args.output_dir = root / "training"
    print("=== Full mixed hierarchical: train Policy/Q/V then held-out games ===")
    print(
        f"train: iterations={args.iterations}, "
        f"games/iteration={args.train_games_per_iteration}, "
        f"simulations={args.train_simulations}"
    )
    print(
        f"eval: games/mode={args.eval_games}, "
        f"simulations={args.eval_simulations}, "
        f"seeds={args.eval_start_seed}.."
        f"{args.eval_start_seed + args.eval_games - 1}"
    )
    records = iterate(training_args)
    final_policy, final_q, final_v = final_checkpoints(
        records, args.candidate_value_checkpoint, args.state_value_checkpoint
    )
    configurations = {
        "before_training": (
            str(args.candidate_policy_checkpoint),
            str(args.candidate_value_checkpoint),
            str(args.state_value_checkpoint) if args.state_value_checkpoint else None,
        ),
        "after_training": (final_policy, final_q, final_v),
    }
    summaries = {}
    device = resolve_device(args.device)
    for name, (policy, q, v) in configurations.items():
        agent_args = SimpleNamespace(**vars(args))
        agent_args.candidate_policy_checkpoint = policy
        agent_args.candidate_value_checkpoint = q
        agent_args.state_value_checkpoint = v
        agent_args.simulations = args.eval_simulations
        agent = agent_factory(agent_args, device)
        output_dir = root / "evaluation" / name
        print(f"--- held-out evaluation: {name} ---")
        summary = collect(
            agent,
            state_factory,
            games=args.eval_games,
            start_seed=args.eval_start_seed,
            max_turns=args.max_turns,
            output_dir=output_dir,
            return_dataset=None,
            gamma=args.gamma,
        )
        if summary.saved_games != args.eval_games:
            raise RuntimeError(f"{name}: incomplete evaluation collection.")
        paths = [
            output_dir / f"rollout_seed_{seed}.npz"
            for seed in range(args.eval_start_seed, args.eval_start_seed + args.eval_games)
        ]
        summaries[name] = report(paths)
        print(
            f"{name}: completed={summaries[name]['completed_games']} "
            f"score_delta={summaries[name]['score_delta']}"
        )
    result = {
        "note": (
            "Held-out simplified two-object-ball games; completion is NOT win rate. "
            "Training returns are only from completed games. "
            "This measures integrated execution, not general billiards strength."
        ),
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "evaluated_checkpoints": {
            name: {
                "policy": policy, "candidate_q": q, "state_v": v
            }
            for name, (policy, q, v) in configurations.items()
        },
        "training": [
            {
                "index": record.index,
                "visited_roots": record.visited_roots,
                "candidate_rows": record.candidate_rows,
                "complete_games": record.complete_games,
                "policy": str(record.output_policy),
                "candidate_q": str(record.output_q) if record.output_q else None,
                "state_v": str(record.output_v) if record.output_v else None,
            }
            for record in records
        ],
        "evaluation": summaries,
    }
    output_path = root / "experiment_summary.json"
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Saved integrated experiment summary: {output_path}")
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--residual-checkpoint", required=True)
    p.add_argument("--candidate-policy-checkpoint", required=True)
    p.add_argument("--candidate-value-checkpoint", required=True)
    p.add_argument("--state-value-checkpoint")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--train-games-per-iteration", type=int, default=256)
    p.add_argument("--eval-games", type=int, default=128)
    p.add_argument("--start-seed", type=int, default=110000)
    p.add_argument("--eval-start-seed", type=int, default=210000)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--train-simulations", type=int, default=50)
    p.add_argument("--eval-simulations", type=int, default=50)
    p.add_argument("--policy-epochs", type=int, default=10)
    p.add_argument("--critic-epochs", type=int, default=10)
    p.add_argument("--update-critics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--training-seed", type=int, default=110000)
    p.add_argument("--root-min-visits", type=int, default=1)
    p.add_argument("--root-tie-breaker", choices=("first", "empirical_q"), default="empirical_q")
    p.add_argument("--candidate-q-prior-visits", type=float, default=2.0)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--residual-scale", type=float, default=0.75)
    p.add_argument("--radius", type=float, default=0.10)
    p.add_argument("--outcome-grid-size", type=int, default=5)
    p.add_argument("--outcome-top-k", type=int, default=8)
    p.add_argument("--outcome-dedup-distance", type=float, default=0.10)
    p.add_argument("--safety-top-k", type=int, default=4)
    p.add_argument("--safety-powers", type=float, nargs="+", default=[0.65, 1.10])
    p.add_argument("--safety-angles", type=float, nargs="+", default=[-12.0, 0.0, 12.0])
    p.add_argument("--safety-tip-y-mm", type=float, nargs="+", default=[0.0, 5.0])
    return p.parse_args(argv)


if __name__ == "__main__":
    run_experiment(parse_args())
