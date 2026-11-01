import argparse

import numpy as np
import torch
import torch.nn as nn

from scripts.build_macro_candidate_value_dataset import (
    random_state,
    resolve_device,
)
from scripts.eval_macro_candidate_value import (
    make_components,
)
from src.env.micro_billiard_env import (
    MicroBilliardEnv,
)
from src.goal.types import (
    TacticalGoalType,
)
from src.macro.candidate import (
    MacroCandidate,
)
from src.macro.candidate_policy import (
    load_macro_candidate_policy_network,
)
from src.macro.candidate_value import (
    load_macro_candidate_value_network,
)
from src.macro.candidate_value_targets import (
    evaluate_two_step_candidate_target,
)
from src.macro.mcts import MCTS


class UniformCandidatePolicy(nn.Module):

    def forward(
        self,
        state,
        candidates,
    ):
        return torch.zeros(
            candidates.shape[0],
            dtype=torch.float32,
            device=candidates.device,
        )


class ZeroValue(nn.Module):

    def forward(
        self,
        state,
    ):
        return torch.zeros(
            1,
            dtype=torch.float32,
            device=state.device,
        )


class StaticCandidateGenerator:

    def __init__(
        self,
        candidates,
    ):
        self.candidates = list(
            candidates
        )

    def generate(
        self,
        state,
    ):
        return list(
            self.candidates
        )


def evaluate_selection(
    selected,
    candidates,
    target_q,
):
    target_q = np.asarray(
        target_q,
        dtype=np.float32,
    )

    selected_index = None

    for index, candidate in enumerate(
        candidates
    ):
        if candidate == selected:
            selected_index = index
            break

    if selected_index is None:
        raise RuntimeError(
            "Selected candidate was not found "
            "in the root candidate list."
        )

    best_q = float(
        np.max(
            target_q
        )
    )
    selected_q = float(
        target_q[
            selected_index
        ]
    )

    return {
        "selected_q": selected_q,
        "best_q": best_q,
        "regret": float(
            best_q
            - selected_q
        ),
        "optimal": float(
            np.isclose(
                selected_q,
                best_q,
                atol=1e-6,
            )
        ),
        "position_selected": float(
            selected.goal.goal_type
            == TacticalGoalType.POSITION_ATTACK
        ),
    }


def summarize(
    records,
):
    if not records:
        raise ValueError(
            "records must not be empty."
        )

    return {
        "optimal_rate": float(
            np.mean(
                [
                    record["optimal"]
                    for record in records
                ]
            )
        ),
        "mean_regret": float(
            np.mean(
                [
                    record["regret"]
                    for record in records
                ]
            )
        ),
        "mean_selected_q": float(
            np.mean(
                [
                    record["selected_q"]
                    for record in records
                ]
            )
        ),
        "position_selection_rate": float(
            np.mean(
                [
                    record[
                        "position_selected"
                    ]
                    for record in records
                ]
            )
        ),
    }


def build_mcts(
    policy_network,
    candidate_value_network,
    root_candidates,
    tree_generator,
    transition,
    args,
    simulations,
):
    return MCTS(
        policy_network=policy_network,
        value_network=ZeroValue(),
        action_generator=tree_generator,
        root_action_generator=(
            StaticCandidateGenerator(
                root_candidates
            )
        ),
        transition_model=transition,
        simulations=simulations,
        c_puct=args.c_puct,
        gamma=args.gamma,
        device="cpu",
        candidate_mode=True,
        force_evaluate_unvisited=False,
        root_min_visits_before_puct=0,
        candidate_value_network=(
            candidate_value_network
        ),
        candidate_q_prior_visits=(
            args.candidate_q_prior_visits
        ),
    )


def resolve_simulation_budgets(
    simulations,
    simulation_budgets,
):
    if simulation_budgets:
        budgets = [
            int(value)
            for value
            in simulation_budgets
        ]
    else:
        budgets = [
            int(simulations)
        ]

    if any(
        value < 1
        for value in budgets
    ):
        raise ValueError(
            "simulation budgets must be >= 1."
        )

    # Preserve user order while removing duplicates.
    return list(
        dict.fromkeys(
            budgets
        )
    )


def main(
    args,
):
    micro_device = resolve_device(
        args.device
    )

    (
        candidate_policy,
        policy_checkpoint,
    ) = (
        load_macro_candidate_policy_network(
            args.candidate_policy_checkpoint,
            device="cpu",
        )
    )

    (
        candidate_value,
        value_checkpoint,
    ) = (
        load_macro_candidate_value_network(
            args.candidate_value_checkpoint,
            device="cpu",
        )
    )

    (
        transition,
        root_generator,
        tree_generator,
    ) = make_components(
        args,
        micro_device,
    )

    sample_env = MicroBilliardEnv()

    simulation_budgets = (
        resolve_simulation_budgets(
            args.simulations,
            args.simulation_budgets,
        )
    )

    uniform_records = {
        budget: []
        for budget
        in simulation_budgets
    }
    learned_records = {
        budget: []
        for budget
        in simulation_budgets
    }

    evaluated_states = 0

    for state_offset in range(
        args.states
    ):
        state = random_state(
            sample_env,
            args.start_seed
            + state_offset,
        )

        root_candidates = list(
            root_generator.generate(
                state
            )
        )

        if not root_candidates:
            continue

        target_q = []

        for candidate in root_candidates:
            target = (
                evaluate_two_step_candidate_target(
                    state=state,
                    candidate=candidate,
                    transition_model=transition,
                    next_candidate_generator=(
                        tree_generator
                    ),
                    gamma=args.gamma,
                )
            )

            target_q.append(
                target.target
            )

        target_q = np.asarray(
            target_q,
            dtype=np.float32,
        )

        for budget in simulation_budgets:
            uniform_mcts = build_mcts(
                policy_network=(
                    UniformCandidatePolicy()
                ),
                candidate_value_network=(
                    candidate_value
                ),
                root_candidates=(
                    root_candidates
                ),
                tree_generator=(
                    tree_generator
                ),
                transition=transition,
                args=args,
                simulations=budget,
            )

            learned_mcts = build_mcts(
                policy_network=(
                    candidate_policy
                ),
                candidate_value_network=(
                    candidate_value
                ),
                root_candidates=(
                    root_candidates
                ),
                tree_generator=(
                    tree_generator
                ),
                transition=transition,
                args=args,
                simulations=budget,
            )

            uniform_selected = (
                uniform_mcts.select_action(
                    state
                )
            )

            learned_selected = (
                learned_mcts.select_action(
                    state
                )
            )

            if not isinstance(
                uniform_selected,
                MacroCandidate,
            ):
                raise RuntimeError(
                    "Uniform MCTS did not select "
                    "a MacroCandidate."
                )

            if not isinstance(
                learned_selected,
                MacroCandidate,
            ):
                raise RuntimeError(
                    "Learned-policy MCTS did not select "
                    "a MacroCandidate."
                )

            uniform_records[
                budget
            ].append(
                evaluate_selection(
                    uniform_selected,
                    root_candidates,
                    target_q,
                )
            )

            learned_records[
                budget
            ].append(
                evaluate_selection(
                    learned_selected,
                    root_candidates,
                    target_q,
                )
            )

        evaluated_states += 1

        if (
            args.log_interval > 0
            and evaluated_states
            % args.log_interval
            == 0
        ):
            print(
                f"states evaluated: "
                f"{evaluated_states}/"
                f"{args.states}"
            )

    if evaluated_states == 0:
        raise RuntimeError(
            "No states with Macro candidates "
            "were evaluated."
        )

    print()
    print(
        "=== Macro search unseen comparison ==="
    )
    print(
        f"micro device: {micro_device}"
    )
    print(
        f"states evaluated: "
        f"{evaluated_states}"
    )
    print(
        "simulation budgets:",
        simulation_budgets,
    )
    print(
        "Candidate Q checkpoint best val MSE:",
        value_checkpoint.get(
            "best_val_mse",
            "n/a",
        ),
    )
    print(
        "Candidate Policy checkpoint best val top-1:",
        policy_checkpoint.get(
            "best_val_metrics",
            {},
        ).get(
            "top1_accuracy",
            "n/a",
        ),
    )

    for budget in simulation_budgets:
        uniform = summarize(
            uniform_records[
                budget
            ]
        )
        learned = summarize(
            learned_records[
                budget
            ]
        )

        print()
        print(
            f"--- simulations={budget} ---"
        )
        print(
            "uniform Policy + learned Q"
        )
        print(
            f"  optimal selection rate: "
            f"{uniform['optimal_rate']:.3f}"
        )
        print(
            f"  mean selected-Q regret: "
            f"{uniform['mean_regret']:.4f}"
        )
        print(
            f"  mean selected target Q: "
            f"{uniform['mean_selected_q']:.4f}"
        )
        print(
            f"  POSITION selection rate: "
            f"{uniform['position_selection_rate']:.3f}"
        )

        print(
            "learned Policy + learned Q"
        )
        print(
            f"  optimal selection rate: "
            f"{learned['optimal_rate']:.3f}"
        )
        print(
            f"  mean selected-Q regret: "
            f"{learned['mean_regret']:.4f}"
        )
        print(
            f"  mean selected target Q: "
            f"{learned['mean_selected_q']:.4f}"
        )
        print(
            f"  POSITION selection rate: "
            f"{learned['position_selection_rate']:.3f}"
        )

        print(
            "learned - uniform"
        )
        print(
            "  optimal selection rate: "
            f"{learned['optimal_rate'] - uniform['optimal_rate']:+.3f}"
        )
        print(
            "  mean regret: "
            f"{learned['mean_regret'] - uniform['mean_regret']:+.4f}"
        )
        print(
            "  mean selected target Q: "
            f"{learned['mean_selected_q'] - uniform['mean_selected_q']:+.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--residual-checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--candidate-policy-checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--candidate-value-checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--states",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=90000,
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=50,
        help=(
            "Single search budget used when "
            "--simulation-budgets is omitted."
        ),
    )
    parser.add_argument(
        "--simulation-budgets",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Evaluate multiple MCTS budgets "
            "on the same unseen states."
        ),
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--candidate-q-prior-visits",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--outcome-grid-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--outcome-top-k",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--outcome-dedup-distance",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    args = parser.parse_args()

    if args.states < 1:
        parser.error(
            "--states must be >= 1"
        )

    if args.simulations < 1:
        parser.error(
            "--simulations must be >= 1"
        )

    if (
        args.simulation_budgets
        is not None
        and any(
            value < 1
            for value
            in args.simulation_budgets
        )
    ):
        parser.error(
            "--simulation-budgets values "
            "must be >= 1"
        )

    if args.candidate_q_prior_visits < 0.0:
        parser.error(
            "--candidate-q-prior-visits "
            "must be >= 0"
        )

    return args


if __name__ == "__main__":
    main(
        parse_args()
    )
