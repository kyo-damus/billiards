import random

import numpy as np

from src.common.types import MacroAction, Strategy
from src.env.fastfiz_simulator import FastFizSimulator
from src.env.micro_billiard_env import MicroBilliardEnv
from src.env.shot_geometry import ghost_ball_position


POWERS = [
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
]


def sample_feasible_case(
    env,
    seed,
    max_attempts=200,
):
    rng = random.Random(seed)

    for attempt in range(max_attempts):
        macro_action = MacroAction(
            strategy=Strategy.ATTACK,
            target_ball=rng.choice([0, 1]),
            target_pocket=rng.choice(range(6)),
        )

        env.set_macro_action(macro_action)

        env.reset(
            seed=seed + attempt
        )

        if env.is_current_macro_feasible():
            raw_state = env.get_state()

            return raw_state.copy(), macro_action

    raise RuntimeError(
        "Could not generate feasible case."
    )


def get_target_position(state, target_ball):
    if target_ball == 0:
        return state[2:4]

    return state[4:6]


def calculate_ideal_angle(
    state,
    macro_action,
    env,
):
    cue_pos = state[0:2]

    target_pos = get_target_position(
        state,
        macro_action.target_ball,
    )

    pocket_pos = env.POCKET_POSITIONS[
        macro_action.target_pocket
    ]

    ghost_pos = ghost_ball_position(
        target_pos,
        pocket_pos,
    )

    direction = ghost_pos - cue_pos

    return float(
        np.degrees(
            np.arctan2(
                direction[1],
                direction[0],
            )
        )
    )


def main():
    num_cases = 200

    env = MicroBilliardEnv()

    success_any_power = 0
    pocketed_anywhere = 0

    success_by_power = {
        power: 0
        for power in POWERS
    }

    for case_index in range(num_cases):
        state, macro_action = sample_feasible_case(
            env,
            seed=1000 + case_index * 100,
        )

        angle = calculate_ideal_angle(
            state,
            macro_action,
            env,
        )

        case_success = False
        case_any_pocket = False

        for power in POWERS:
            sim = FastFizSimulator()

            sim.set_state(state)

            sim.step(
                target_ball=macro_action.target_ball,
                power=power,
                angle=angle,
            )

            pocket_index = sim.get_pocket_index(
                macro_action.target_ball
            )

            if pocket_index != -1:
                case_any_pocket = True

            if (
                pocket_index
                == macro_action.target_pocket
            ):
                case_success = True
                success_by_power[power] += 1

        success_any_power += int(case_success)
        pocketed_anywhere += int(case_any_pocket)

    print()
    print("=== Direct Shot Baseline ===")
    print(f"cases: {num_cases}")

    print(
        "success with any power: "
        f"{success_any_power / num_cases:.3f}"
    )

    print(
        "pocketed anywhere: "
        f"{pocketed_anywhere / num_cases:.3f}"
    )

    print()
    print("Success rate by physical power:")

    for power in POWERS:
        rate = (
            success_by_power[power]
            / num_cases
        )

        print(
            f"  power={power:3.1f}: "
            f"{rate:.3f}"
        )


if __name__ == "__main__":
    main()