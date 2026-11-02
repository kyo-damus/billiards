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
from src.common.types import GameState
from src.game.simple_rules import SimplifiedGameRules
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


def classify_safety_after_elevated_shot(
    original_state, snapshot, first_contact_ball, intended_ball,
    rules=None,
):
    """Apply simplified game rules to an elevation-probe result.

    Return a next opponent-turn GameState only for a v0-eligible
    nonpotting Safety. This does not validate official foul rules.
    """
    if first_contact_ball != intended_ball:
        return None
    before = np.asarray(
        original_state.ball_pocket_indices, dtype=np.int32
    )
    after = np.asarray(snapshot.pocket_indices, dtype=np.int32)
    if after[0] >= 0:
        return None
    if np.any((before[1:] < 0) & (after[1:] >= 0)):
        return None

    physical = GameState(
        ball_positions=np.asarray(
            snapshot.positions, dtype=np.float32
        ).copy(),
        ball_pocket_indices=after.copy(),
        score=original_state.score.copy(),
        current_player=original_state.current_player,
    )
    transition = (
        rules if rules is not None else SimplifiedGameRules()
    ).apply(
        previous_state=original_state,
        physical_state=physical,
        shot_info={"scratched": False},
    )
    if transition.terminated or not transition.turn_changed:
        return None
    return transition.next_state


def probe_elevation_grid(
    state,
    elevations,
    angle_offsets,
    power=1.0,
    powers=None,
    tip_y_mm=(0.0,),
    threat_evaluator=None,
    threat_samples_per_elevation=0,
):
    """Return per-elevation FastFiz feasibility and first-contact counts.

    Use the same angle/power/tip grid at EVERY elevation.
    The single-power argument remains for backwards compatibility.
    No opponent lookahead or learned Policy is involved.
    """
    powers = (power,) if powers is None else tuple(powers)
    tips = tuple(tip_y_mm)
    if threat_samples_per_elevation < 0:
        raise ValueError("threat_samples_per_elevation must be >= 0.")
    if threat_samples_per_elevation and threat_evaluator is None:
        raise ValueError("Threat evaluator required for threat sampling.")
    if (
        not powers
        or not tips
        or any(not np.isfinite(v) or not 0.5 <= v <= 3.0 for v in powers)
        or any(not np.isfinite(v) or abs(v) > 12.0 for v in tips)
    ):
        raise ValueError("Invalid power/tip diagnostic grid.")
    sim = FastFizSimulator()
    rules = SimplifiedGameRules()
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
        any_first = 0
        safety_like = 0
        first_contact_example = None
        eligible_safety = 0
        sampled_threat_count = 0
        minimum_opponent_direct_success = None

        for target in target_balls:
            for offset in angle_offsets:
                for current_power in powers:
                    for current_tip_y in tips:
                        shot = plan_defense_shot(
                            state,
                            target_ball=target,
                            power=current_power,
                            angle_offset_deg=offset,
                            tip_x_mm=0.0,
                            tip_y_mm=current_tip_y,
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
                            tip_x_mm=shot.tip_x_mm,
                            tip_y_mm=shot.tip_y_mm,
                            cue_elevation_deg=elevation,
                        )
                        attempted += 1
                        codes[result.precondition_code] += 1
                        placement_codes[result.placement_code] += 1
                        if result.precondition_code != 0:
                            continue

                        valid += 1
                        if result.first_contact_ball >= 0:
                            any_first += 1
                        if result.first_contact_ball != target:
                            continue
                        correct_first += 1
                        if first_contact_example is None:
                            first_contact_example = {
                                "target": target,
                                "angle_offset": float(offset),
                                "power": float(current_power),
                                "tip_y_mm": float(current_tip_y),
                                "elevation": float(elevation),
                            }

                        snapshot = sim.snapshot()
                        scratched = snapshot.pocket_indices[0] >= 0
                        new_pots = np.any(
                            (state.ball_pocket_indices[1:] < 0)
                            & (snapshot.pocket_indices[1:] >= 0)
                        )
                        if not scratched and not new_pots:
                            safety_like += 1

                        next_state = classify_safety_after_elevated_shot(
                            state,
                            snapshot,
                            result.first_contact_ball,
                            target,
                            rules=rules,
                        )
                        if next_state is not None:
                            eligible_safety += 1
                            if (
                                threat_evaluator is not None
                                and sampled_threat_count
                                < threat_samples_per_elevation
                            ):
                                threat = threat_evaluator.evaluate(
                                    next_state,
                                    defending_player=state.current_player,
                                )
                                count = int(threat.successful_pot_count)
                                minimum_opponent_direct_success = (
                                    count
                                    if minimum_opponent_direct_success is None
                                    else min(
                                        minimum_opponent_direct_success, count
                                    )
                                )
                                sampled_threat_count += 1

        rows.append({
            "elevation": float(elevation),
            "attempted": attempted,
            "valid": valid,
            "correct_first": correct_first,
            "any_first": any_first,
            "safety_like": safety_like,
            "eligible_safety": eligible_safety,
            "sampled_threat_count": sampled_threat_count,
            "min_opponent_direct_success": minimum_opponent_direct_success,
            "first_contact_example": first_contact_example,
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
    print(f"powers: {args.powers if args.powers is not None else [args.power]}")
    print(f"tip_y_mm: {args.tip_y_mm}")
    print(
        f"opponent DIRECT threat samples per elevation: "
        f"{args.threat_samples_per_elevation}"
    )
    if args.threat_samples_per_elevation:
        print(
            "Threat is deterministic under the existing 10-degree "
            "opponent Micro; samples follow shot-grid order, "
            "not a representative win-rate estimate."
        )

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
                powers=args.powers,
                tip_y_mm=args.tip_y_mm,
                threat_evaluator=(
                    safety_shots.opponent_threat_evaluator
                    if args.threat_samples_per_elevation else None
                ),
                threat_samples_per_elevation=(
                    args.threat_samples_per_elevation
                ),
            ):
                print(
                    f"  theta={row['elevation']:4.0f} deg "
                    f"valid={row['valid']}/{row['attempted']} "
                    f"any_first={row['any_first']} "
                    f"correct_first={row['correct_first']} "
                    f"no_pot_no_scratch_after_contact={row['safety_like']} "
                    f"simplified_safety_eligible={row['eligible_safety']} "
                    f"sampled_opponent_threat={row['sampled_threat_count']} "
                    f"min_opponent_DIRECT_success="
                    f"{row['min_opponent_direct_success']} "
                    f"codes=[{summarize_code_counts(row['codes'])}]"
                )
                if row["first_contact_example"] is not None:
                    print(
                        "    first_correct_contact_example="
                        f"{row['first_contact_example']}"
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
        "--powers", type=float, nargs="+",
        help="Optional power grid; overrides --power.",
    )
    parser.add_argument(
        "--tip-y-mm", type=float, nargs="+", default=[0.0],
    )
    parser.add_argument(
        "--threat-samples-per-elevation", type=int, default=0,
        help=(
            "Evaluate up to N eligible elevated shots per elevation "
            "using the CURRENT fixed-elevation opponent DIRECT Micro."
        ),
    )
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
        or (
            args.powers is not None
            and (
                not args.powers
                or any(
                    not np.isfinite(v) or not 0.5 <= v <= 3.0
                    for v in args.powers
                )
            )
        )
        or args.threat_samples_per_elevation < 0
        or not args.tip_y_mm
        or any(
            not np.isfinite(v) or abs(v) > 12.0
            for v in args.tip_y_mm
        )
    ):
        parser.error("Invalid elevation/shot grid configuration.")
    return args


if __name__ == "__main__":
    main(parse_args())
