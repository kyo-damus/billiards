"""End-to-end smoke for physical DEFENSE MacroCandidate transition.

This does not use the pretrained Attack Candidate Q or Policy to rank
Defense. It verifies generation -> representation -> physical transition.
"""

import argparse

import numpy as np

from scripts.build_macro_candidate_value_dataset import (
    random_state,
    resolve_device,
)
from scripts.eval_macro_candidate_value import make_components
from src.env.micro_billiard_env import MicroBilliardEnv
from src.macro.defense_transition import (
    DefenseMacroTransitionModel,
    MixedStrategyMacroTransitionModel,
)
from src.macro.opponent_threat import (
    OpponentDirectThreatEvaluator,
)
from src.macro.outcome_guided_safety_candidates import (
    OutcomeGuidedSafetyMacroCandidateGenerator,
)
from src.macro.outcome_guided_safety_shots import (
    OutcomeGuidedSafetyShotGenerator,
)
from src.macro.candidate_encoding import (
    encode_macro_candidate_v2,
)
from src.micro.defense_shot import DefenseShotExecutor


def main(args):
    device = resolve_device(args.device)

    attack_transition, _, direct_generator = make_components(
        args,
        device,
    )
    defense_executor = DefenseShotExecutor()
    threat_evaluator = OpponentDirectThreatEvaluator(
        transition_model=attack_transition,
        candidate_generator=direct_generator,
    )
    physical_generator = OutcomeGuidedSafetyShotGenerator(
        defense_executor=defense_executor,
        opponent_threat_evaluator=threat_evaluator,
        top_k=args.top_k,
    )

    # Reuse the already evaluated proposals; don't repeat
    # expensive FastFiz opponent lookahead for representation.
    defense_generator = OutcomeGuidedSafetyMacroCandidateGenerator(
        shot_generator=physical_generator,
    )
    mixed = MixedStrategyMacroTransitionModel(
        attack_transition_model=attack_transition,
        defense_transition_model=DefenseMacroTransitionModel(
            executor=defense_executor,
        ),
    )

    env = MicroBilliardEnv()
    with_safety = 0
    physically_consistent = 0
    total_defense = 0
    attack_executed = 0

    print("=== Mixed MacroTransition smoke ===")
    print(f"device: {device}")
    print(f"boards: {args.states}")

    for i in range(args.states):
        current = random_state(
            env,
            args.start_seed + i,
        )
        attack_candidates = direct_generator.generate(
            current
        )

        if attack_candidates:
            attack_result = mixed.step(
                current,
                attack_candidates[0],
            )
            attack_executed += 1
            print(
                f"board={i} attack_reward={attack_result.reward:.3f}"
                f" attack_next_player="
                f"{attack_result.next_state.current_player}"
            )

        # Generator performs full physical/threat search on this
        # board, then converts selected proposals into candidates.
        physical_proposals = physical_generator.generate(
            current
        )
        defense_candidates = defense_generator.from_proposals(
            physical_proposals
        )

        if not defense_candidates:
            print(
                f"board={i} no SAFETY candidates"
            )
            continue

        with_safety += 1
        candidate = defense_candidates[0]

        if encode_macro_candidate_v2(candidate).shape != (31,):
            raise RuntimeError(
                "Defense encoding must have 31 features."
            )

        result = mixed.step(current, candidate)
        total_defense += 1

        # Replay the command independently from the same board
        # and match it to the previously evaluated proposal.
        original = next(
            (
                proposal
                for proposal in physical_proposals
                if (
                    proposal.targeted_ball
                    == candidate.action.target_ball
                    and np.isclose(
                        proposal.shot.angle,
                        candidate.physical_shot.angle,
                    )
                    and np.isclose(
                        proposal.shot.power,
                        candidate.physical_shot.power,
                    )
                    and np.isclose(
                        proposal.shot.tip_x_mm,
                        candidate.physical_shot.tip_x_mm,
                    )
                    and np.isclose(
                        proposal.shot.tip_y_mm,
                        candidate.physical_shot.tip_y_mm,
                    )
                )
            ),
            None,
        )

        if original is not None and np.allclose(
            result.next_state.ball_positions,
            original.result.next_state.ball_positions,
            atol=1e-5,
        ):
            physically_consistent += 1

        evaluation = result.evaluation
        print(
            f"board={i} safety_reward={result.reward:.3f}"
            f" goal_achieved={evaluation.goal_achieved}"
            f" eligible={evaluation.safety_eligible}"
            f" first_contact={evaluation.first_contact_ball}"
            f" next_player={result.next_state.current_player}"
        )

    print()
    print(f"Attack transitions executed: {attack_executed}")
    print(f"Boards with Safety candidates: {with_safety}/{args.states}")
    print(f"Safety transitions executed: {total_defense}")
    print(
        "Safety physical replay consistent: "
        f"{physically_consistent}/{total_defense}"
    )
    print(
        "This is an execution contract smoke test, "
        "not an Attack-vs-Defense policy comparison."
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-checkpoint",
        required=True,
    )
    parser.add_argument(
        "--residual-checkpoint",
        required=True,
    )
    parser.add_argument(
        "--states", type=int, default=2
    )
    parser.add_argument(
        "--start-seed", type=int, default=90000
    )
    parser.add_argument(
        "--top-k", type=int, default=4
    )
    parser.add_argument(
        "--residual-scale", type=float, default=0.75
    )
    parser.add_argument(
        "--radius", type=float, default=0.10
    )
    parser.add_argument(
        "--outcome-grid-size", type=int, default=5
    )
    parser.add_argument(
        "--outcome-top-k", type=int, default=8
    )
    parser.add_argument(
        "--outcome-dedup-distance", type=float, default=0.10
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    args = parser.parse_args()

    if args.states < 1 or args.top_k < 1:
        parser.error("--states and --top-k must be >= 1")

    return args


if __name__ == "__main__":
    main(parse_args())
