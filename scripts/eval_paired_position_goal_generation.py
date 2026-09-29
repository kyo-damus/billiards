import argparse
from dataclasses import dataclass

import numpy as np
import torch

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)
from src.env.micro_billiard_env import (
    MicroBilliardEnv,
)
from src.goal.reward import (
    calculate_goal_conditioned_reward,
)
from src.macro.outcome_guided_position_generator import (
    OutcomeGuidedPositionCandidateGenerator,
)
from src.macro.position_candidate_generator import (
    PositionAttackCandidateGenerator,
)
from src.micro.goal_conditioned_sac import (
    GoalConditionedSACAgent,
)


@dataclass
class EvalResult:
    pot: bool
    cue_region: bool
    position_goal: bool
    any_pocket: bool
    wrong_pocket: bool
    scratch: bool
    cue_distance: float
    reward: float


def load_agent(
    checkpoint_path,
    device,
):
    agent = GoalConditionedSACAgent(
        action_dim=2,
        device=device,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    agent.agent.actor.load_state_dict(
        checkpoint["actor"]
    )

    agent.agent.critic.load_state_dict(
        checkpoint["critic"]
    )

    if "critic_target" in checkpoint:
        agent.agent.critic_target.load_state_dict(
            checkpoint["critic_target"]
        )

    return agent


def build_game_state(
    env,
):
    snapshot = env.sim.snapshot()

    return GameState(
        ball_positions=(
            snapshot.positions.copy()
        ),
        score=np.zeros(
            2,
            dtype=np.float32,
        ),
        current_player=0,
        ball_pocket_indices=(
            snapshot.pocket_indices.copy()
        ),
    )


def sample_state(
    env,
    seed,
):
    # reset用の仮Action
    env.set_macro_action(
        MacroAction(
            strategy=Strategy.ATTACK,
            target_ball=0,
            target_pocket=0,
        )
    )

    env.reset(
        seed=seed
    )

    return build_game_state(
        env
    )


def candidate_key(
    candidate,
):
    """
    cue target以外の戦術条件を表すkey。

    Paired比較ではこのkeyを揃え、
    Ideal / Outcomeでcue target生成方法だけを変える。
    """

    goal = candidate.goal

    return (
        candidate.action.target_ball,
        candidate.action.target_pocket,
        goal.next_target_ball,
        goal.next_target_pocket,
    )


def build_pairs(
    state,
    ideal_generator,
    outcome_generator,
):
    """
    同一GameStateについて、
    current target/pocketとnext target/pocketが
    完全一致するIdeal / Outcome候補を対応付ける。

    Outcome側で同じkeyが複数ある場合は、
    generatorがnext_shot_distance順に返すため
    最初の候補を採用する。
    """

    ideal_candidates = (
        ideal_generator.generate(
            state
        )
    )

    outcome_witnesses = (
        outcome_generator
        .generate_with_witnesses(
            state
        )
    )

    ideal_by_key = {
        candidate_key(candidate):
        candidate
        for candidate in ideal_candidates
    }

    outcome_by_key = {}

    for witness in outcome_witnesses:
        key = candidate_key(
            witness.candidate
        )

        if key not in outcome_by_key:
            outcome_by_key[
                key
            ] = witness.candidate

    common_keys = sorted(
        set(ideal_by_key)
        & set(outcome_by_key)
    )

    return [
        (
            ideal_by_key[key],
            outcome_by_key[key],
        )
        for key in common_keys
    ]


def evaluate_candidate(
    env,
    agent,
    state,
    candidate,
):
    env.reset_to_game_state(
        state,
        candidate.action,
    )

    observation = (
        env.get_physical_observation()
    )

    action = agent.select_action(
        observation,
        candidate.goal,
        deterministic=True,
    )

    (
        _,
        base_reward,
        _,
        _,
        info,
    ) = env.step(
        action
    )

    next_observation = (
        env.get_physical_observation()
    )

    (
        reward,
        goal_info,
    ) = calculate_goal_conditioned_reward(
        base_reward=base_reward,
        goal=candidate.goal,
        before_observation=observation,
        after_observation=next_observation,
        shot_info=info,
    )

    cue_distance = float(
        goal_info.get(
            "cue_distance_after",
            float("nan"),
        )
    )

    return EvalResult(
        pot=bool(
            info.get(
                "success",
                False,
            )
        ),
        cue_region=bool(
            goal_info.get(
                "cue_region_reached",
                False,
            )
        ),
        position_goal=bool(
            goal_info.get(
                "goal_achieved",
                False,
            )
        ),
        any_pocket=bool(
            info.get(
                "target_pocketed_anywhere",
                False,
            )
        ),
        wrong_pocket=bool(
            info.get(
                "wrong_pocket",
                False,
            )
        ),
        scratch=bool(
            info.get(
                "scratched",
                False,
            )
        ),
        cue_distance=cue_distance,
        reward=float(
            reward
        ),
    )


def summarize(
    results,
):
    n = len(results)

    if n == 0:
        raise RuntimeError(
            "No evaluation results."
        )

    distances = [
        result.cue_distance
        for result in results
        if np.isfinite(
            result.cue_distance
        )
    ]

    return {
        "n": n,
        "pot": float(
            np.mean(
                [
                    result.pot
                    for result in results
                ]
            )
        ),
        "cue_region": float(
            np.mean(
                [
                    result.cue_region
                    for result in results
                ]
            )
        ),
        "position_goal": float(
            np.mean(
                [
                    result.position_goal
                    for result in results
                ]
            )
        ),
        "any_pocket": float(
            np.mean(
                [
                    result.any_pocket
                    for result in results
                ]
            )
        ),
        "wrong_pocket": float(
            np.mean(
                [
                    result.wrong_pocket
                    for result in results
                ]
            )
        ),
        "scratch": float(
            np.mean(
                [
                    result.scratch
                    for result in results
                ]
            )
        ),
        "mean_cue_distance": (
            float(
                np.mean(
                    distances
                )
            )
            if distances
            else float("nan")
        ),
        "median_cue_distance": (
            float(
                np.median(
                    distances
                )
            )
            if distances
            else float("nan")
        ),
        "reward": float(
            np.mean(
                [
                    result.reward
                    for result in results
                ]
            )
        ),
    }


def print_summary(
    name,
    summary,
):
    print()
    print(
        f"[{name}]"
    )
    print(
        f"pairs: "
        f"{summary['n']}"
    )
    print(
        "position pot: "
        f"{summary['pot']:.3f}"
    )
    print(
        "cue region: "
        f"{summary['cue_region']:.3f}"
    )
    print(
        "position goal: "
        f"{summary['position_goal']:.3f}"
    )
    print(
        "any pocket: "
        f"{summary['any_pocket']:.3f}"
    )
    print(
        "wrong pocket: "
        f"{summary['wrong_pocket']:.3f}"
    )
    print(
        "scratch: "
        f"{summary['scratch']:.3f}"
    )
    print(
        "mean cue distance: "
        f"{summary['mean_cue_distance']:.3f} m"
    )
    print(
        "median cue distance: "
        f"{summary['median_cue_distance']:.3f} m"
    )
    print(
        "avg goal reward: "
        f"{summary['reward']:.3f}"
    )


def main(
    args,
):
    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    print(
        f"device: {device}"
    )
    print(
        f"checkpoint: "
        f"{args.checkpoint}"
    )

    agent = load_agent(
        args.checkpoint,
        device,
    )

    state_env = (
        MicroBilliardEnv()
    )

    eval_env = (
        MicroBilliardEnv()
    )

    ideal_generator = (
        PositionAttackCandidateGenerator(
            target_radius=args.radius
        )
    )

    outcome_generator = (
        OutcomeGuidedPositionCandidateGenerator(
            grid_size=(
                args.outcome_grid_size
            ),
            target_radius=(
                args.radius
            ),
            top_k=(
                args.outcome_top_k
            ),
            dedup_distance=(
                args.outcome_dedup_distance
            ),
        )
    )

    rng = np.random.default_rng(
        args.seed
    )

    ideal_results = []
    outcome_results = []

    matched_states = 0
    attempts = 0

    outcome_only_goal = 0
    ideal_only_goal = 0
    both_goal = 0
    neither_goal = 0

    while (
        len(ideal_results) < args.pairs
        and attempts
        < args.pairs * 50
    ):
        attempts += 1

        state = sample_state(
            state_env,
            seed=(
                args.seed
                + attempts * 1000
            ),
        )

        pairs = build_pairs(
            state,
            ideal_generator,
            outcome_generator,
        )

        if not pairs:
            continue

        matched_states += 1

        # 1盤面が多数のpairで結果を支配しないよう、
        # 各盤面から1pairだけランダムに採用する。
        pair_index = int(
            rng.integers(
                0,
                len(pairs),
            )
        )

        (
            ideal_candidate,
            outcome_candidate,
        ) = pairs[
            pair_index
        ]

        if (
            ideal_candidate.action
            != outcome_candidate.action
        ):
            raise RuntimeError(
                "Paired candidates must "
                "share the same MacroAction."
            )

        if (
            ideal_candidate.goal
            .next_target_ball
            != outcome_candidate.goal
            .next_target_ball
            or ideal_candidate.goal
            .next_target_pocket
            != outcome_candidate.goal
            .next_target_pocket
        ):
            raise RuntimeError(
                "Paired candidates must "
                "share the same next shot."
            )

        ideal_result = (
            evaluate_candidate(
                eval_env,
                agent,
                state,
                ideal_candidate,
            )
        )

        outcome_result = (
            evaluate_candidate(
                eval_env,
                agent,
                state,
                outcome_candidate,
            )
        )

        ideal_results.append(
            ideal_result
        )
        outcome_results.append(
            outcome_result
        )

        if (
            outcome_result.position_goal
            and not ideal_result.position_goal
        ):
            outcome_only_goal += 1

        elif (
            ideal_result.position_goal
            and not outcome_result.position_goal
        ):
            ideal_only_goal += 1

        elif (
            ideal_result.position_goal
            and outcome_result.position_goal
        ):
            both_goal += 1

        else:
            neither_goal += 1

        pair_no = len(
            ideal_results
        )

        if (
            pair_no % args.log_interval
            == 0
        ):
            print(
                f"pairs={pair_no:4d} "
                f"matched_states={matched_states:4d} "
                f"ideal_goal="
                f"{np.mean([x.position_goal for x in ideal_results]):.3f} "
                f"outcome_goal="
                f"{np.mean([x.position_goal for x in outcome_results]):.3f}"
            )

    if not ideal_results:
        raise RuntimeError(
            "No matched Ideal/Outcome pairs."
        )

    if len(
        ideal_results
    ) < args.pairs:
        print(
            "warning: requested "
            f"{args.pairs} pairs but only "
            f"{len(ideal_results)} were found."
        )

    ideal_summary = summarize(
        ideal_results
    )

    outcome_summary = summarize(
        outcome_results
    )

    print()
    print(
        "=== Paired POSITION Goal Generation Evaluation ==="
    )
    print(
        f"requested pairs: "
        f"{args.pairs}"
    )
    print(
        f"evaluated pairs: "
        f"{len(ideal_results)}"
    )
    print(
        f"matched states: "
        f"{matched_states}"
    )
    print(
        f"target radius: "
        f"{args.radius:.3f} m"
    )
    print(
        "same current MacroAction: yes"
    )
    print(
        "same next target/pocket: yes"
    )

    print_summary(
        "Ideal cue target",
        ideal_summary,
    )

    print_summary(
        "Outcome-guided cue target",
        outcome_summary,
    )

    print()
    print(
        "[Paired POSITION Goal outcome]"
    )
    print(
        "outcome-only success: "
        f"{outcome_only_goal / len(ideal_results):.3f}"
    )
    print(
        "ideal-only success: "
        f"{ideal_only_goal / len(ideal_results):.3f}"
    )
    print(
        "both success: "
        f"{both_goal / len(ideal_results):.3f}"
    )
    print(
        "neither success: "
        f"{neither_goal / len(ideal_results):.3f}"
    )
    print(
        "position-goal difference: "
        f"{outcome_summary['position_goal'] - ideal_summary['position_goal']:+.3f}"
    )
    print(
        "cue-region difference: "
        f"{outcome_summary['cue_region'] - ideal_summary['cue_region']:+.3f}"
    )
    print(
        "pot difference: "
        f"{outcome_summary['pot'] - ideal_summary['pot']:+.3f}"
    )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--pairs",
        type=int,
        default=200,
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
        default=25,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=10000,
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

    return parser.parse_args()


if __name__ == "__main__":
    main(
        parse_args()
    )
