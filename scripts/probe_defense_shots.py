"""Small FastFiz smoke for explicit physical Defense shots.

No Defense learned policy or Macro/MCTS integration is assumed.
"""

import argparse

import numpy as np

from scripts.build_macro_candidate_value_dataset import (
    random_state,
)
from src.env.micro_billiard_env import MicroBilliardEnv
from src.micro.defense_shot import (
    DefenseShotExecutor,
    plan_defense_shot,
)


def main(args):
    env = MicroBilliardEnv()
    executor = DefenseShotExecutor()

    total = 0
    valid = 0
    invalid = 0
    changed_turn = 0
    scratched = 0
    object_moved = 0
    new_pots = 0
    first_contacts = 0
    intended_first_contacts = 0
    examples = []

    for offset in range(args.boards):
        state = random_state(
            env,
            args.start_seed + offset,
        )

        for ball in (0, 1):
            for power in args.powers:
                for angle_offset in args.offsets:
                    total += 1
                    shot = plan_defense_shot(
                        state,
                        target_ball=ball,
                        power=power,
                        angle_offset_deg=angle_offset,
                        tip_x_mm=args.tip_x_mm,
                        tip_y_mm=args.tip_y_mm,
                    )

                    try:
                        result = executor.execute(
                            state,
                            target_ball=ball,
                            shot=shot,
                        )
                    except RuntimeError as error:
                        invalid += 1
                        if len(examples) < 3:
                            examples.append(
                                f"invalid board={offset} "
                                f"ball={ball} power={power} "
                                f"offset={angle_offset} "
                                f"reason={error}"
                            )
                        continue

                    valid += 1
                    changed_turn += int(
                        result.turn_changed
                    )
                    scratched += int(
                        result.scratched
                    )
                    object_moved += int(
                        result.object_ball_moved
                    )
                    new_pots += (
                        result.newly_pocketed_object_count
                    )
                    first_contacts += int(
                        result.first_contact_ball >= 0
                    )
                    intended_first_contacts += int(
                        result.first_contact_verified
                    )

                    if len(examples) < 3:
                        examples.append(
                            f"board={offset} ball={ball} "
                            f"power={power} offset={angle_offset} "
                            f"angle={shot.angle:.1f} "
                            f"turn_changed={result.turn_changed} "
                            f"scratch={result.scratched} "
                            f"object_moved={result.object_ball_moved} "
                            f"first_contact={result.first_contact_ball}"
                        )

    print("=== Defense physical shot probe ===")
    print(f"boards: {args.boards}")
    print(f"attempts: {total}")
    print(f"valid FastFiz shots: {valid}")
    print(f"rejected shots: {invalid}")
    print(f"turn changes: {changed_turn}")
    print(f"scratches: {scratched}")
    print(f"shots moving an object ball: {object_moved}")
    print(f"new object ball pockets: {new_pots}")
    print(f"shots contacting any object ball first: {first_contacts}")
    print(f"shots contacting intended object first: {intended_first_contacts}")
    print(
        "First cue/object collision is observed. "
        "Full foul legality is not checked."
    )

    for item in examples:
        print(item)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--boards",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=90000,
    )
    parser.add_argument(
        "--powers",
        type=float,
        nargs="+",
        default=[0.65, 1.10],
    )
    parser.add_argument(
        "--offsets",
        type=float,
        nargs="+",
        default=[-12.0, 0.0, 12.0],
    )
    parser.add_argument(
        "--tip-x-mm",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--tip-y-mm",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    if args.boards < 1:
        parser.error("--boards must be >= 1")
    if any(
        not np.isfinite(value)
        or not 0.5 <= value <= 3.0
        for value in args.powers
    ):
        parser.error("--powers must be in [0.5, 3.0]")
    if any(
        not np.isfinite(value)
        or abs(value) > 35.0
        for value in args.offsets
    ):
        parser.error("--offsets must be in [-35, 35]")
    if (
        not np.isfinite(args.tip_x_mm)
        or not np.isfinite(args.tip_y_mm)
        or args.tip_x_mm ** 2
        + args.tip_y_mm ** 2 > 144.0
    ):
        parser.error("tip offset radius must be <= 12 mm")
    return args


if __name__ == "__main__":
    main(parse_args())
