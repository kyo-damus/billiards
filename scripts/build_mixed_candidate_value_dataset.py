"""Build a 31-D mixed Attack/Safety two-step Candidate Q dataset.

Attack roots: DIRECT/POSITION. Safety roots: physically executable,
outcome-guided selections (NOT learned Defense Micro).

A *common* turn-aware two-step target is used for both:

    Q_target = immediate_reward +
               gamma * sign(first_turn, next_turn) *
               max(next_DIRECT_immediate_reward)

The next-step action set intentionally contains only DIRECT shots to
keep the target cheap and consistent between Attack and Safety.
Selection of Safety root candidates already uses full deterministic
opponent DIRECT lookahead, so the dataset is selection-biased. These
targets are neither full-game values nor evidence of generalization.

Never overwrite the Attack-v0 (25-D) dataset.
"""

import argparse
from pathlib import Path

import numpy as np

from scripts.build_macro_candidate_value_dataset import (
    GOAL_TYPE_TO_ID,
    random_state,
    resolve_device,
)
from scripts.eval_macro_candidate_value import (
    make_components,
)
from src.common.types import Strategy
from src.env.micro_billiard_env import MicroBilliardEnv
from src.macro.candidate_encoding import (
    MACRO_CANDIDATE_V2_DIM,
    encode_macro_candidate_v2,
)
from src.macro.candidate_value_targets import (
    evaluate_two_step_candidate_target,
)
from src.macro.defense_transition import (
    DefenseMacroTransitionModel,
    MixedStrategyMacroTransitionModel,
)
from src.macro.encoding import encode_game_state
from src.macro.opponent_threat import OpponentDirectThreatEvaluator
from src.macro.outcome_guided_safety_candidates import (
    OutcomeGuidedSafetyMacroCandidateGenerator,
)
from src.macro.outcome_guided_safety_shots import (
    OutcomeGuidedSafetyShotGenerator,
)
from src.micro.defense_shot import DefenseShotExecutor


def make_mixed_components(args, device):
    attack, attack_generator, direct_generator = make_components(
        args, device
    )

    defense_executor = DefenseShotExecutor()
    defense_transition = DefenseMacroTransitionModel(
        executor=defense_executor,
    )
    transition = MixedStrategyMacroTransitionModel(
        attack_transition_model=attack,
        defense_transition_model=defense_transition,
    )
    threat = OpponentDirectThreatEvaluator(
        transition_model=attack,
        candidate_generator=direct_generator,
    )
    shot_generator = OutcomeGuidedSafetyShotGenerator(
        defense_executor=defense_executor,
        opponent_threat_evaluator=threat,
        powers=args.safety_powers,
        angle_offsets_deg=args.safety_angles,
        tip_offsets_mm=[
            (0.0, float(tip_y))
            for tip_y in args.safety_tip_y_mm
        ],
        top_k=args.safety_top_k,
    )
    candidate_adapter = OutcomeGuidedSafetyMacroCandidateGenerator(
        shot_generator=shot_generator,
    )

    return (
        transition,
        attack_generator,
        direct_generator,
        shot_generator,
        candidate_adapter,
    )


def build_dataset(args):
    device = resolve_device(args.device)
    (
        transition,
        attack_generator,
        direct_generator,
        safety_shots,
        safety_adapter,
    ) = make_mixed_components(args, device)

    state_env = MicroBilliardEnv()

    encoded_states = []
    encoded_candidates = []
    q_targets = []
    rewards = []
    next_rewards = []
    next_candidate_counts = []
    state_ids = []
    goal_types = []
    strategies = []
    board_count = 0
    attack_count = 0
    safety_count = 0
    shot_attempts = 0
    eligible_shots = 0

    for board in range(args.states):
        state = random_state(
            state_env,
            args.seed + board,
        )

        attacks = list(
            attack_generator.generate(state)
        )
        evaluated_shots = safety_shots.generate(state)
        shot_attempts += safety_shots.last_stats["attempted"]
        eligible_shots += safety_shots.last_stats["eligible"]

        safeties = safety_adapter.from_proposals(
            evaluated_shots
        )
        candidates = attacks + safeties

        if not candidates:
            continue

        board_id = board_count
        board_count += 1
        encoded_state = encode_game_state(state).astype(np.float32)

        for candidate in candidates:
            label = evaluate_two_step_candidate_target(
                state=state,
                candidate=candidate,
                transition_model=transition,
                next_candidate_generator=direct_generator,
                gamma=args.gamma,
                turn_aware=True,
            )

            candidate_encoding = encode_macro_candidate_v2(
                candidate
            )

            if candidate_encoding.shape != (
                MACRO_CANDIDATE_V2_DIM,
            ):
                raise RuntimeError("Unexpected candidate v2 dimension.")

            encoded_states.append(encoded_state.copy())
            encoded_candidates.append(
                candidate_encoding.astype(np.float32)
            )
            q_targets.append(label.target)
            rewards.append(label.immediate_reward)
            next_rewards.append(label.next_best_reward)
            next_candidate_counts.append(label.next_candidate_count)
            state_ids.append(board_id)
            goal_types.append(
                GOAL_TYPE_TO_ID[candidate.goal.goal_type]
            )
            is_safety = (
                candidate.action.strategy == Strategy.DEFENSE
            )
            strategies.append(1 if is_safety else 0)
            safety_count += int(is_safety)
            attack_count += int(not is_safety)

        if args.log_interval > 0 and (
            board + 1
        ) % args.log_interval == 0:
            print(
                f"boards processed: {board + 1}/{args.states}, "
                f"samples: {len(q_targets)}, Safety: {safety_count}"
            )

    if not q_targets:
        raise RuntimeError("No mixed-strategy Candidate Q samples.")

    if attack_count == 0 or safety_count == 0:
        raise RuntimeError(
            "Dataset must contain BOTH Attack and Safety. "
            "Increase --states or enlarge the physical shot grid."
        )

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        state=np.stack(encoded_states).astype(np.float32),
        candidate=np.stack(encoded_candidates).astype(np.float32),
        q_target=np.asarray(q_targets, dtype=np.float32),
        immediate_reward=np.asarray(rewards, dtype=np.float32),
        next_best_reward=np.asarray(next_rewards, dtype=np.float32),
        next_candidate_count=np.asarray(
            next_candidate_counts, dtype=np.int32
        ),
        state_id=np.asarray(state_ids, dtype=np.int32),
        goal_type=np.asarray(goal_types, dtype=np.int32),
        strategy=np.asarray(strategies, dtype=np.int32),
        candidate_dim=np.asarray(
            MACRO_CANDIDATE_V2_DIM, dtype=np.int32
        ),
        candidate_encoding_version=np.asarray(2, dtype=np.int32),
        gamma=np.asarray(args.gamma, dtype=np.float32),
        residual_scale=np.asarray(
            args.residual_scale, dtype=np.float32
        ),
        turn_aware=np.asarray(True, dtype=np.bool_),
        next_action_set=np.asarray("DIRECT_ONLY"),
        safety_selection=np.asarray("OUTCOME_GUIDED"),
        seed_start=np.asarray(args.seed, dtype=np.int64),
        requested_states=np.asarray(args.states, dtype=np.int32),
    )

    q = np.asarray(q_targets, dtype=np.float32)
    print()
    print("=== Mixed-strategy Candidate Q dataset v2 ===")
    print(f"device: {device}")
    print(f"boards requested / kept: {args.states} / {board_count}")
    print(f"samples total / Attack / Safety: {len(q)} / {attack_count} / {safety_count}")
    print(f"physical Safety proposals / eligible: {shot_attempts} / {eligible_shots}")
    print(f"candidate dimension: {MACRO_CANDIDATE_V2_DIM}")
    print("turn-aware: True; next-step set: DIRECT_ONLY")
    print(f"target mean/std: {q.mean():.4f} / {q.std():.4f}")
    print(f"target min/max: {q.min():.4f} / {q.max():.4f}")
    print(f"output: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument(
        "--output", default="macro_candidate_value_mixed_v2.npz"
    )
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument("--seed", type=int, default=93000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--residual-scale", type=float, default=0.75)
    parser.add_argument("--radius", type=float, default=0.10)
    parser.add_argument("--outcome-grid-size", type=int, default=5)
    parser.add_argument("--outcome-top-k", type=int, default=8)
    parser.add_argument("--outcome-dedup-distance", type=float, default=0.10)
    parser.add_argument("--safety-top-k", type=int, default=4)
    parser.add_argument(
        "--safety-powers", nargs="+", type=float,
        default=[0.65, 1.10],
    )
    parser.add_argument(
        "--safety-angles", nargs="+", type=float,
        default=[-12.0, 0.0, 12.0],
    )
    parser.add_argument(
        "--safety-tip-y-mm", nargs="+", type=float,
        default=[0.0, 5.0],
    )
    args = parser.parse_args()

    if args.states < 1:
        parser.error("--states must be >= 1")
    if args.safety_top_k < 1:
        parser.error("--safety-top-k must be >= 1")
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must be in [0,1]")
    if (
        not args.safety_powers
        or not all(
            np.isfinite(v) and 0.5 <= v <= 3.0
            for v in args.safety_powers
        )
    ):
        parser.error("safety powers must be in [0.5, 3.0]")
    if (
        not args.safety_angles
        or not all(
            np.isfinite(v) and abs(v) <= 35.0
            for v in args.safety_angles
        )
    ):
        parser.error("safety angles must be in [-35, 35]")
    if (
        not args.safety_tip_y_mm
        or not all(
            np.isfinite(v) and abs(v) <= 12.0
            for v in args.safety_tip_y_mm
        )
    ):
        parser.error("safety tip y must be in [-12, 12]")

    return args


if __name__ == "__main__":
    build_dataset(parse_args())
