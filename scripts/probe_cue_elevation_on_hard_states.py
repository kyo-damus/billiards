"""Isolated FastFiz cue-elevation feasibility probe on post-Safety boards.

This DOES NOT alter the 10-degree action semantics of the trained
Attack Micro, executable Defense MacroCandidates, or 31D Candidate Q.

For reproducibility, derive hard states from the existing SAFETY
generator. For each, evaluate native precondition codes and first
collision at several cue elevations on identical shot directions.
Also record whether ATTACK actions actually passed the native
precondition, which older Attack success counts did not distinguish.

Only a physical *diagnostic*. A shot at a different elevation
cannot be executed by the current MacroCandidate/31D Q model.
"""

import argparse
from collections import Counter

import numpy as np

from scripts.build_macro_candidate_value_dataset import (
    random_state,
    resolve_device,
)
from scripts.build_mixed_candidate_value_dataset import (
    make_mixed_components,
)
from scripts.eval_post_safety_hard_states import (
    select_hard_states_from_safety,
)
from src.common.table import TABLE_LENGTH, TABLE_WIDTH
from src.env.fastfiz_simulator import (
    FastFizSimulator,
    describe_fastfiz_precondition,
)
from src.env.micro_billiard_env import MicroBilliardEnv
from src.env.simulator_state import SimulatorSnapshot
from src.micro.defense_shot import plan_defense_shot


def rail_distances(positions):
    x, y = (float(v) for v in positions[0])
    return {
        "west": x,
        "east": TABLE_WIDTH - x,
        "south": y,
        "north": TABLE_LENGTH - y,
    }


def probe_elevation_grid(state, elevations, angle_offsets, power=1.0):
    """Return per-elevation preconditions and cue-contact outcomes.

    Identical shot directions, power and tip positions are used for
    all elevations. No opponent-threat lookahead is performed.
    """
    sim = FastFizSimulator()
    rows = []

    target_balls = [
        ball for ball in (0, 1)
        if not state.is_pocketed(ball + 1)
    ]

    for elevation in elevations:
        codes = Counter()
        placement_codes = Counter()
        attempted = 0
        valid = 0
        correct_first = 0
        safety_like = 0

        for target in target_balls:
            for offset in angle_offsets:
                shot = plan_defense_shot(
                    state,
                    target_ball=target,
                    power=power,
                    angle_offset_deg=offset,
                    tip_x_mm=0.0,
                    tip_y_mm=0.0,
                )
                sim.restore(
                    SimulatorSnapshot(
                        positions=np.asarray(
                            state.ball_positions, dtype=np.float32
                        ).copy(),
                        pocket_indices=np.asarray(
                            state.ball_pocket_indices, dtype=np.int32
                        ).copy(),
                    )
                )
                result = sim.step(
                    target_ball=target,
                    power=shot.power,
                    angle=shot.angle,
                    cue_elevation_deg=elevation,
                )
                attempted += 1
                codes[result.precondition_code] += 1
                placement_codes[result.placement_code] += 1
                if result.precondition_code != 0:
                    continue

                valid += 1
                if result.first_contact_ball != target:
                    continue
                correct_first += 1

                snapshot = sim.snapshot()
                scratched = snapshot.pocket_indices[0] >= 0
                new_pots = np.any(
                    (state.ball_pocket_indices[1:] < 0)
                    & (snapshot.pocket_indices[1:] >= 0)
                )
                if not scratched and not new_pots:
                    safety_like += 1

        rows.append({
            "elevation": float(elevation),
            "attempted": attempted,
            "valid": valid,
            "correct_first": correct_first,
            "safety_like": safety_like,
            "codes": codes,
            "placements": placement_codes,
        })

    return rows


def attack_precondition_summary(state, attack_candidates, transition):
    """Count physics-rejected Attack shots separately from misses."""
    attack_model = transition.attack_transition_model
    if not hasattr(attack_model, "last_shot_info"):
        raise RuntimeError(
            "Attack transition must expose last_shot_info."
        )
    counts = Counter()
    for candidate in attack_candidates:
        result = attack_model.step(state, candidate)
        info = attack_model.last_shot_info
        if info is None:
            raise RuntimeError("Attack shot diagnostics are unavailable.")
        if not info.get("physically_possible", False):
            counts["physics_invalid"] += 1
            counts[f"precondition_{info['precondition_code']}"] += 1
        elif result.evaluation is not None and (
            result.evaluation.target_potted
            and not result.evaluation.scratch
        ):
            counts["valid_pot"] += 1
        else:
            counts["valid_no_pot"] += 1
    return counts


def summarize_code_counts(counts):
    return ", ".join(
        f"{code}:{describe_fastfiz_precondition(code)}={n}"
        for code, n in sorted(counts.items())
    )


def main(args):
    device = resolve_device(args.device)
    (
        transition,
        attack_generator,
        _,
        safety_shots,
        _,
    ) = make_mixed_components(args, device)

    seen = set()
    env = MicroBilliardEnv()
    examined_states = 0

    print("=== FastFiz cue-elevation / rail collision diagnostic ===")
    print(
        "Not a change to learned Micro/31D candidate representation."
    )
    print(f"device: {device}")
    print(f"source boards: {args.source_boards}")
    print(f"elevations: {args.elevations}")
    print(f"angles: {args.angle_offsets}")

    for board in range(args.source_boards):
        initial = random_state(env, args.source_seed + board)
        proposals = safety_shots.generate(initial)
        hard_states = select_hard_states_from_safety(
            initial,
            proposals,
            max_successful_direct=args.max_successful_direct,
            seen=seen,
        )

        for sample in hard_states:
            state = sample.state
            distances = rail_distances(state.ball_positions)
            nearest = min(distances, key=distances.get)
            attacks = list(attack_generator.generate(state))
            attack_counts = attack_precondition_summary(
                state, attacks, transition
            )

            print(
                f"source_board={board}, cue_xy="
                f"{np.round(state.ball_positions[0], 5).tolist()}, "
                f"nearest_rail={nearest}, "
                f"distance={distances[nearest]:.4f}m"
            )
            print(
                f"  ATTACK n={len(attacks)}: "
                f"physics_invalid={attack_counts['physics_invalid']}, "
                f"valid_pot={attack_counts['valid_pot']}, "
                f"valid_no_pot={attack_counts['valid_no_pot']}"
            )
            code_str = summarize_code_counts({
                int(key.removeprefix("precondition_")): value
                for key, value in attack_counts.items()
                if key.startswith("precondition_")
            })
            if code_str:
                print(f"  ATTACK invalid codes: {code_str}")

            for row in probe_elevation_grid(
                state,
                elevations=args.elevations,
                angle_offsets=args.angle_offsets,
                power=args.power,
            ):
                print(
                    f"  theta={row['elevation']:4.0f} deg "
                    f"valid={row['valid']}/{row['attempted']} "
                    f"correct_first={row['correct_first']} "
                    f"no_pot_no_scratch_after_contact={row['safety_like']} "
                    f"codes=[{summarize_code_counts(row['codes'])}]"
                )

            examined_states += 1
            if examined_states >= args.max_hard_states:
                break

        if examined_states >= args.max_hard_states:
            break

    print(
        f"hard states examined: {examined_states}; "
        "only precondition/contact sensitivity is established."
    )
    if examined_states == 0:
        print("No qualifying post-Safety states in the source scan.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--source-boards", type=int, default=5)
    parser.add_argument("--source-seed", type=int, default=90000)
    parser.add_argument("--max-hard-states", type=int, default=2)
    parser.add_argument("--max-successful-direct", type=int, default=0)
    parser.add_argument(
        "--elevations",
        nargs="+", type=float,
        default=[10.0, 20.0, 35.0, 50.0, 60.0],
    )
    parser.add_argument(
        "--angle-offsets",
        nargs="+", type=float,
        default=[-20.0, -10.0, 0.0, 10.0, 20.0],
    )
    parser.add_argument("--power", type=float, default=1.0)
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
    if (
        args.source_boards < 1
        or args.max_hard_states < 1
        or args.max_successful_direct < 0
        or not args.elevations
        or any(
            not np.isfinite(x) or not 0.0 <= x <= 70.0
            for x in args.elevations
        )
        or not args.angle_offsets
        or any(
            not np.isfinite(x) or abs(x) > 35.0
            for x in args.angle_offsets
        )
        or not np.isfinite(args.power)
        or not 0.5 <= args.power <= 3.0
    ):
        parser.error("Invalid elevation/shot grid configuration.")
    return args


if __name__ == "__main__":
    main(parse_args())
