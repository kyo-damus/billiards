"""Probe Attack-difficult opponent-turn states produced by physical SAFETY.

These are *selected* post-SAFETY states from deterministic FastFiz.
Selection is outcome-conditioned: do NOT treat them as an unbiased
sample of naturally encountered gameplay or proof of Defense value.

First find post-SAFETY states with zero successful DIRECT responses,
then compare EVERY Attack and Safety root candidate by the SAME
turn-aware two-step teacher. Optionally compare MCTS choices too.
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
from scripts.eval_mixed_mcts_difficult_boards import (
    assess_attack_feasibility,
    chosen_from_root,
    compare_teacher_oracles,
    make_search,
    teacher_value,
)
from src.common.types import Strategy
from src.env.micro_billiard_env import MicroBilliardEnv
from src.macro.candidate_value import (
    load_macro_candidate_value_network,
)


@dataclass(frozen=True)
class SourcedHardState:
    state: object
    source_target_ball: int
    source_direct_successes: int
    source_shot_angle: float
    source_shot_power: float
    source_tip_y_mm: float


def board_fingerprint(state, bin_size=0.03):
    """Coarse duplicate screen, not a physics or state identity test."""
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0.")

    positions = np.asarray(state.ball_positions, dtype=np.float64)
    pockets = np.asarray(state.ball_pocket_indices, dtype=np.int64)
    if positions.shape != (3, 2) or pockets.shape != (3,):
        raise ValueError("Invalid GameState geometry/pockets.")

    return (
        int(state.current_player),
        tuple(pockets.tolist()),
        tuple(np.rint(positions.ravel() / bin_size).astype(int)),
    )


def select_hard_states_from_safety(
    previous_state,
    proposals,
    max_successful_direct=0,
    seen=None,
):
    """
    Reuse already evaluated Safety proposals (no extra physical
    search), keeping poststates with few executable opponent DIRECTs.
    """
    if max_successful_direct < 0:
        raise ValueError("max_successful_direct must be >= 0.")
    if seen is None:
        seen = set()

    selected = []
    for proposal in proposals:
        state = proposal.result.next_state
        if (
            state.current_player == previous_state.current_player
            or state.is_pocketed(0)
        ):
            continue
        successes = int(
            proposal.opponent_threat.successful_pot_count
        )
        if successes > max_successful_direct:
            continue
        key = board_fingerprint(state)
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            SourcedHardState(
                state=state,
                source_target_ball=int(proposal.targeted_ball),
                source_direct_successes=successes,
                source_shot_angle=float(proposal.shot.angle),
                source_shot_power=float(proposal.shot.power),
                source_tip_y_mm=float(proposal.shot.tip_y_mm),
            )
        )

    return selected


def main(args):
    device = resolve_device(args.device)
    (
        transition,
        attack_generator,
        direct_generator,
        safety_shots,
        safety_adapter,
    ) = make_mixed_components(args, device)

    q_net = None
    if args.run_mcts:
        if not args.candidate_value_checkpoint:
            raise ValueError(
                "--candidate-value-checkpoint is required with --run-mcts."
            )
        q_net, checkpoint = load_macro_candidate_value_network(
            args.candidate_value_checkpoint,
            device=device,
            expected_candidate_dim=31,
        )
        if (
            int(checkpoint.get("candidate_encoding_version", -1)) != 2
            or not bool(checkpoint.get("turn_aware", False))
        ):
            raise ValueError("Expected 31D turn-aware mixed Q v2.")

    initial_env = MicroBilliardEnv()
    seen = set()
    source_boards = 0
    selected_hard_states = 0
    tested_hard_states = 0
    no_attack = 0
    filtered_attack = 0
    no_safety = 0
    oracle_deltas = []
    mcts_safety_chosen = 0
    mcts_deltas = []

    print("=== Post-SAFETY hard-state teacher oracle ===")
    print(
        "Selected via deterministic physical Safety search. "
        "This is a targeted stress test, not unbiased gameplay."
    )
    print(f"device: {device}")
    print(
        f"source seeds: {args.source_seed}.."
        f"{args.source_seed + args.source_boards - 1}"
    )
    print(
        f"post-Safety DIRECT success threshold: "
        f"{args.max_successful_direct}"
    )
    print(
        f"subsequent all-Attack success threshold: "
        f"{args.max_successful_attacks}"
    )
    print("second-step teacher action set: DIRECT_ONLY")

    for board in range(args.source_boards):
        initial = random_state(
            initial_env,
            args.source_seed + board,
        )
        source_boards += 1
        proposals = safety_shots.generate(initial)
        sourced = select_hard_states_from_safety(
            initial,
            proposals,
            max_successful_direct=args.max_successful_direct,
            seen=seen,
        )
        selected_hard_states += len(sourced)

        for sample in sourced:
            hard = sample.state
            attack_choices = list(
                attack_generator.generate(hard)
            )
            if not attack_choices:
                no_attack += 1
                continue

            feasibility = assess_attack_feasibility(
                hard, attack_choices, transition
            )
            if (
                feasibility.successful_pot_count
                > args.max_successful_attacks
            ):
                filtered_attack += 1
                continue

            new_safety_proposals = safety_shots.generate(hard)
            safety_choices = safety_adapter.from_proposals(
                new_safety_proposals
            )
            if not safety_choices:
                no_safety += 1
                continue

            oracle = compare_teacher_oracles(
                state=hard,
                attack_candidates=attack_choices,
                safety_candidates=safety_choices,
                transition=transition,
                direct_generator=direct_generator,
                gamma=args.gamma,
            )

            oracle_deltas.append(oracle.safety_advantage)
            tested_hard_states += 1

            mcts_suffix = ""
            if args.run_mcts:
                root_attack = make_search(
                    args, device, q_net, transition,
                    attack_choices, direct_generator,
                ).search(hard)
                root_mixed = make_search(
                    args, device, q_net, transition,
                    attack_choices + safety_choices,
                    direct_generator,
                ).search(hard)

                a_choice, _ = chosen_from_root(root_attack)
                mixed_choice, _ = chosen_from_root(root_mixed)
                mcts_safety_chosen += int(
                    mixed_choice.action.strategy == Strategy.DEFENSE
                )
                delta = (
                    teacher_value(
                        hard, mixed_choice, transition,
                        direct_generator, args.gamma
                    )
                    - teacher_value(
                        hard, a_choice, transition,
                        direct_generator, args.gamma
                    )
                )
                mcts_deltas.append(delta)
                mcts_suffix = (
                    f" selected={mixed_choice.goal.goal_type.value}"
                    f" chosen_delta={delta:+.3f}"
                )

            print(
                f"source_board={board}"
                f" source_opponent_DIRECT_success={sample.source_direct_successes}"
                f" root_Attack_success={feasibility.successful_pot_count}/"
                f"{feasibility.candidate_count}"
                f" root_DIRECT_success={feasibility.successful_direct_count}"
                f" choices_Attack/Safety={len(attack_choices)}/"
                f"{len(safety_choices)}"
                f" oracle_Attack={oracle.best_attack:+.3f}"
                f" oracle_Safety={oracle.best_safety:+.3f}"
                f" oracle_Safety_adv={oracle.safety_advantage:+.3f}"
                f"{mcts_suffix}"
            )
            if tested_hard_states >= args.max_compared:
                break

        if tested_hard_states >= args.max_compared:
            break

    print()
    print(
        f"source boards / unique post-Safety hard states / "
        f"paired comparisons: {source_boards}/"
        f"{selected_hard_states}/{tested_hard_states}"
    )
    print(
        f"no Attack candidates: {no_attack}; "
        f"filtered by all-Attack success: {filtered_attack}; "
        f"no Safety continuation: {no_safety}"
    )

    if oracle_deltas:
        a = np.asarray(oracle_deltas, dtype=np.float64)
        print(
            "oracle Safety better/tied/worse: "
            f"{int((a > 1e-6).sum())}/"
            f"{int((np.abs(a) <= 1e-6).sum())}/"
            f"{int((a < -1e-6).sum())}"
        )
        print(f"mean Safety-minus-Attack teacher Q: {a.mean():+.4f}")
    else:
        print(
            "No paired states met the filters. "
            "This is a coverage limit, not a negative result."
        )

    if args.run_mcts and mcts_deltas:
        print(
            f"mixed MCTS selected Safety: "
            f"{mcts_safety_chosen}/{len(mcts_deltas)}"
        )
        print(
            "mean selected mixed-minus-Attack-only teacher Q: "
            f"{np.mean(mcts_deltas):+.4f}"
        )
    print(
        "Caution: poststates were optimized for opponent difficulty "
        "using the SAME model used for the teacher. No win rate, "
        "generalization or official foul legality is established."
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--candidate-value-checkpoint")
    parser.add_argument("--source-seed", type=int, default=90000)
    parser.add_argument("--source-boards", type=int, default=5)
    parser.add_argument("--max-compared", type=int, default=2)
    parser.add_argument("--max-successful-direct", type=int, default=0)
    parser.add_argument("--max-successful-attacks", type=int, default=2)
    parser.add_argument("--run-mcts", action="store_true")
    parser.add_argument("--simulations", type=int, default=40)
    parser.add_argument("--root-min-visits", type=int, default=1)
    parser.add_argument(
        "--candidate-q-prior-visits", type=float, default=2.0
    )
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.99)
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
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )

    args = parser.parse_args()
    if (
        args.source_boards < 1
        or args.max_compared < 1
        or args.max_successful_direct < 0
        or args.max_successful_attacks < 0
        or args.simulations < 1
        or args.root_min_visits < 0
        or args.candidate_q_prior_visits < 0
        or not 0.0 <= args.gamma <= 1.0
    ):
        parser.error("Invalid board/sample/MCTS configuration.")

    return args


if __name__ == "__main__":
    main(parse_args())
