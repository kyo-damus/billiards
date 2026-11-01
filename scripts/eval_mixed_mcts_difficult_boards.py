"""Paired, difficulty-stratified Attack-only vs Attack+Safety MCTS.

An ATTACK-hard board is defined operationally: the current deterministic
Attack Micro does not pot a target on more than a configurable number
of eligible Attack candidates. This is not a general difficulty label.

Both MCTS searches receive precisely the same Attack candidates, Q
checkpoint, simulations, discount, zero state V and uniform priors.
The mixed search additionally receives executable Safety candidates.
The comparison is within the deterministic training physics model,
not an unbiased estimate of game win rate.
"""

import argparse
from dataclasses import dataclass

import numpy as np

from scripts.build_macro_candidate_value_dataset import (
    random_state,
    resolve_device,
)
from scripts.build_mixed_candidate_value_dataset import (
    make_mixed_components,
)
from src.common.types import Strategy
from src.goal.types import TacticalGoalType
from src.env.micro_billiard_env import MicroBilliardEnv
from src.macro.candidate_value import (
    load_macro_candidate_value_network,
)
from src.macro.candidate_value_targets import (
    evaluate_two_step_candidate_target,
)
from src.macro.mcts import MCTS
from src.macro.mixed_search import (
    FixedCandidateGenerator,
    ZeroValue,
)


@dataclass(frozen=True)
class AttackFeasibility:
    candidate_count: int
    successful_pot_count: int
    best_immediate_reward: float
    successful_direct_count: int = 0
    distinct_successful_routes: int = 0


def assess_attack_feasibility(
    state,
    attack_candidates,
    transition,
) -> AttackFeasibility:
    """Evaluate every Attack candidate on the exact same root board."""
    best_reward = -float("inf")
    potted = 0
    direct_potted = 0
    successful_routes = set()
    total = 0

    for candidate in attack_candidates:
        if candidate.action.strategy != Strategy.ATTACK:
            raise ValueError("Only ATTACK candidates are allowed.")

        result = transition.step(state, candidate)
        total += 1
        best_reward = max(best_reward, float(result.reward))

        evaluation = result.evaluation
        if (
            evaluation is not None
            and bool(evaluation.target_potted)
            and not bool(evaluation.scratch)
        ):
            potted += 1
            if (
                candidate.goal.goal_type
                == TacticalGoalType.DIRECT_ATTACK
            ):
                direct_potted += 1
            # The same target/pocket can be represented by both
            # DIRECT and POSITION goals. This counts successful
            # tactical routes, NOT unique physical stroke outcomes.
            successful_routes.add(
                (
                    candidate.action.target_ball,
                    candidate.action.target_pocket,
                )
            )

    return AttackFeasibility(
        candidate_count=total,
        successful_pot_count=potted,
        best_immediate_reward=(
            best_reward if total > 0 else float("nan")
        ),
        successful_direct_count=direct_potted,
        distinct_successful_routes=len(successful_routes),
    )


def make_search(
    args,
    device,
    value,
    transition,
    root_candidates,
    next_direct_generator,
):
    return MCTS(
        policy_network=None,
        value_network=ZeroValue().to(device),
        action_generator=next_direct_generator,
        root_action_generator=FixedCandidateGenerator(
            root_candidates
        ),
        transition_model=transition,
        simulations=args.simulations,
        c_puct=args.c_puct,
        gamma=args.gamma,
        device=device,
        candidate_mode=True,
        candidate_value_network=value,
        candidate_q_prior_visits=args.candidate_q_prior_visits,
        candidate_encoding_version=2,
        uniform_candidate_priors=True,
        turn_aware=True,
        root_min_visits_before_puct=args.root_min_visits,
    )


def chosen_from_root(root):
    candidate, child = max(
        root.children.items(),
        key=lambda item: item[1].visit_count,
    )
    return candidate, child


def teacher_value(
    state,
    candidate,
    transition,
    direct_generator,
    gamma,
):
    return evaluate_two_step_candidate_target(
        state=state,
        candidate=candidate,
        transition_model=transition,
        next_candidate_generator=direct_generator,
        gamma=gamma,
        turn_aware=True,
    ).target



@dataclass(frozen=True)
class TeacherOracleComparison:
    """Per-board best two-step teacher Q, irrespective of the MCTS ranking."""

    best_attack: float
    best_safety: float

    @property
    def safety_advantage(self) -> float:
        return self.best_safety - self.best_attack


def compare_teacher_oracles(
    state,
    attack_candidates,
    safety_candidates,
    transition,
    direct_generator,
    gamma,
) -> TeacherOracleComparison:
    """
    Evaluate the same target used to train Candidate Q for every
    root candidate. This is expensive and is therefore optional.

    If Safety's oracle Q is below Attack's, Safety is not optimal
    under the current teacher on this board, independent of MCTS.
    """
    if not attack_candidates or not safety_candidates:
        raise ValueError(
            "Oracle comparison requires both Attack and Safety choices."
        )
    if any(
        candidate.action.strategy != Strategy.ATTACK
        for candidate in attack_candidates
    ) or any(
        candidate.action.strategy != Strategy.DEFENSE
        for candidate in safety_candidates
    ):
        raise ValueError(
            "Oracle candidates have mismatched strategies."
        )

    best_attack = max(
        teacher_value(
            state, candidate, transition, direct_generator, gamma
        )
        for candidate in attack_candidates
    )
    best_safety = max(
        teacher_value(
            state, candidate, transition, direct_generator, gamma
        )
        for candidate in safety_candidates
    )
    return TeacherOracleComparison(
        best_attack=float(best_attack),
        best_safety=float(best_safety),
    )

def main(args):
    device = resolve_device(args.device)
    q_net, checkpoint = load_macro_candidate_value_network(
        args.candidate_value_checkpoint,
        device=device,
        expected_candidate_dim=31,
    )
    if (
        int(checkpoint.get("candidate_encoding_version", -1)) != 2
        or not bool(checkpoint.get("turn_aware", False))
    ):
        raise ValueError(
            "Expected a turn-aware mixed Candidate Q v2 checkpoint."
        )

    (
        transition,
        attack_generator,
        direct_generator,
        safety_shots,
        safety_adapter,
    ) = make_mixed_components(args, device)

    env = MicroBilliardEnv()

    examined = 0
    qualified = 0
    compared = 0
    no_safety = 0
    chose_safety = 0
    teacher_deltas = []
    root_attack_visit_counts = []
    root_safety_visit_counts = []
    hardness_histogram = {"0": 0, "1": 0, "2": 0, "3+": 0}
    oracle_advantages = []

    print("=== Paired MCTS on Attack-difficult boards ===")
    print(f"device: {device}")
    print(f"scan boards: {args.scan_boards}")
    print(
        f"hard criterion: at most {args.max_successful_attacks} "
        "successful ATTACK Micro candidates"
    )
    print(
        f"simulations: {args.simulations}, "
        f"root minimum visits: {args.root_min_visits}"
    )
    print(
        "both searches use the same 31-D Q, zero state V, "
        "uniform priors, and DIRECT-only continuations"
    )

    for board in range(args.scan_boards):
        state = random_state(
            env,
            args.start_seed + board,
        )
        examined += 1

        attacks = list(
            attack_generator.generate(state)
        )
        if not attacks:
            continue

        screen = assess_attack_feasibility(
            state,
            attacks,
            transition,
        )
        successes = screen.successful_pot_count
        bucket = str(successes) if successes <= 2 else "3+"
        hardness_histogram[bucket] += 1

        if (
            screen.successful_pot_count
            > args.max_successful_attacks
        ):
            continue

        qualified += 1

        # Generate Safety just once, after the Attack-only screen,
        # and pass the same fixed Attack candidates to both MCTS.
        proposals = safety_shots.generate(state)
        safeties = safety_adapter.from_proposals(
            proposals
        )

        if not safeties:
            no_safety += 1
            print(
                f"board={board} attack_success="
                f"{screen.successful_pot_count}/{screen.candidate_count} "
                f"direct_success={screen.successful_direct_count} "
                f"distinct_routes={screen.distinct_successful_routes} "
                "Safety not found in current physical grid"
            )
            continue

        attack_root = make_search(
            args,
            device,
            q_net,
            transition,
            attacks,
            direct_generator,
        ).search(state)

        mixed_root = make_search(
            args,
            device,
            q_net,
            transition,
            attacks + safeties,
            direct_generator,
        ).search(state)

        attack_choice, attack_child = chosen_from_root(
            attack_root
        )
        mixed_choice, mixed_child = chosen_from_root(
            mixed_root
        )
        compared += 1
        chose_safety += int(
            mixed_choice.action.strategy == Strategy.DEFENSE
        )

        attack_teacher = teacher_value(
            state, attack_choice, transition,
            direct_generator, args.gamma
        )
        mixed_teacher = teacher_value(
            state, mixed_choice, transition,
            direct_generator, args.gamma
        )
        delta = mixed_teacher - attack_teacher
        teacher_deltas.append(delta)

        oracle_text = ""
        if args.oracle_diagnostic:
            oracle = compare_teacher_oracles(
                state=state,
                attack_candidates=attacks,
                safety_candidates=safeties,
                transition=transition,
                direct_generator=direct_generator,
                gamma=args.gamma,
            )
            oracle_advantages.append(oracle.safety_advantage)
            oracle_text = (
                f" oracle_best_attack={oracle.best_attack:+.3f}"
                f" oracle_best_safety={oracle.best_safety:+.3f}"
                f" oracle_safety_advantage={oracle.safety_advantage:+.3f}"
            )

        attack_visits = sum(
            node.visit_count for candidate, node
            in mixed_root.children.items()
            if candidate.action.strategy == Strategy.ATTACK
        )
        safety_visits = sum(
            node.visit_count for candidate, node
            in mixed_root.children.items()
            if candidate.action.strategy == Strategy.DEFENSE
        )
        root_attack_visit_counts.append(attack_visits)
        root_safety_visit_counts.append(safety_visits)

        print(
            f"board={board} attack_success="
            f"{screen.successful_pot_count}/{screen.candidate_count} "
            f"direct_success={screen.successful_direct_count} "
            f"distinct_routes={screen.distinct_successful_routes} "
            f"best_attack_immediate={screen.best_immediate_reward:+.3f} "
            f"root_choices={len(attacks)}/{len(safeties)} "
            f"mixed_visits={attack_visits}/{safety_visits} "
            f"chosen_attack_only={attack_choice.goal.goal_type.value} "
            f"chosen_mixed={mixed_choice.goal.goal_type.value} "
            f"chosen_teacher_Q={attack_teacher:+.3f}/"
            f"{mixed_teacher:+.3f} "
            f"delta={delta:+.3f} "
            f"selected_visits={attack_child.visit_count}/"
            f"{mixed_child.visit_count}"
            f"{oracle_text}"
        )

        if args.simulations < len(attacks) + len(safeties):
            print(
                "  warning: fewer simulations than mixed root "
                "candidates; some branches may be unvisited"
            )

        if compared >= args.max_compared:
            break

    print()
    print(
        f"examined / hard-qualified / paired: "
        f"{examined} / {qualified} / {compared}"
    )
    print(f"hard boards with no Safety proposals: {no_safety}")
    print(
        "Attack success counts among boards with candidates "
        "(0/1/2/3+): "
        f"{hardness_histogram['0']}/{hardness_histogram['1']}/"
        f"{hardness_histogram['2']}/{hardness_histogram['3+']}"
    )

    if not compared:
        print(
            "No paired board found. This is a coverage limitation, "
            "not proof of Attack/Defense superiority. "
            "Try --max-successful-attacks 2 or --scan-boards 50."
        )
        return

    delta_array = np.asarray(teacher_deltas)
    print(f"mixed chose Safety: {chose_safety}/{compared}")
    print(
        "mean mixed-minus-Attack-only two-step teacher Q: "
        f"{delta_array.mean():+.4f}"
    )
    print(
        "mixed better/tied/worse on teacher: "
        f"{int((delta_array > 1e-6).sum())}/"
        f"{int((np.abs(delta_array) <= 1e-6).sum())}/"
        f"{int((delta_array < -1e-6).sum())}"
    )
    print(
        "mean mixed root visits Attack/Safety: "
        f"{np.mean(root_attack_visit_counts):.2f}/"
        f"{np.mean(root_safety_visit_counts):.2f}"
    )
    if oracle_advantages:
        d = np.asarray(oracle_advantages, dtype=np.float64)
        print(
            "Oracle SAFETY beats/ties/loses to best ATTACK: "
            f"{int((d > 1e-6).sum())}/"
            f"{int((np.abs(d) <= 1e-6).sum())}/"
            f"{int((d < -1e-6).sum())}"
        )
        print(
            "Mean oracle SAFETY-minus-ATTACK two-step Q: "
            f"{d.mean():+.4f}"
        )
    print(
        "Caution: screens use the same deterministic Micro and "
        "FastFiz as the Q teacher, and Safety proposals are "
        "in-model outcome-optimized. This is a controlled "
        "search diagnostic, NOT an unbiased gameplay result."
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--candidate-value-checkpoint", required=True)
    parser.add_argument("--start-seed", type=int, default=95000)
    parser.add_argument("--scan-boards", type=int, default=20)
    parser.add_argument("--max-compared", type=int, default=3)
    parser.add_argument("--max-successful-attacks", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=60)
    parser.add_argument(
        "--oracle-diagnostic",
        action="store_true",
        help=(
            "Evaluate EVERY ATTACK/SAFETY root candidate's "
            "two-step teacher Q on paired boards (extra FastFiz cost)."
        ),
    )
    parser.add_argument("--root-min-visits", type=int, default=1)
    parser.add_argument(
        "--candidate-q-prior-visits",
        type=float, default=2.0,
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
    parser.add_argument("--outcome-dedup-distance", type=float, default=0.10)
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
        "--safety-tip-y-mm", type=float, nargs="+",
        default=[0.0, 5.0],
    )
    args = parser.parse_args()

    if (
        args.scan_boards < 1
        or args.max_compared < 1
        or args.max_successful_attacks < 0
        or args.simulations < 1
        or args.root_min_visits < 0
        or args.candidate_q_prior_visits < 0
        or not 0 <= args.gamma <= 1
    ):
        parser.error("Invalid search/discount/threshold argument.")

    return args


if __name__ == "__main__":
    main(parse_args())
