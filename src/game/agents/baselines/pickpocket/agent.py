"""Nine-ball adaptation of PickPocket-style noisy physical-shot sampling.

Source inspiration: Michael Smith, "Running the Table: An AI for
Computer Billiards", AAAI 2006; "PickPocket: A computer billiards shark",
Artificial Intelligence 171 (2007).

Not a faithful PickPocket reproduction: this v0 is root-only Monte Carlo,
uses a much smaller on-the-fly candidate set (no precomputed difficulty
table, banks, kicks, combination shots, or full depth-2 search), and
replaces eight-ball rules with this project's WPA-style nine-ball rules.
It uses the frozen Phase28a Micro **only for the break shot**; all ordinary
shots are geometric physical 5D proposals, NOT the proposed Micro actor.
"""
from dataclasses import dataclass, replace
import hashlib
import math

import numpy as np

from src.game.nine_ball_agents import GeometryNineBallAgent, ShotPlan
from src.game.nine_ball_encoding_v3 import (
    NineBallGoal, direct_pocket_candidates, encode_rack_v3, reference_angle_deg,
)
from src.game.nine_ball_learned_agent import NineBallMicroSACAgent
from src.game.nine_ball_physics_guard import (
    InvalidPhysicalTransition, simulate_nine_ball_shot, simulate_with_cue_clearance,
)
from src.game.nine_ball_rules import NineBallContext, NineBallRules
from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.micro.shot_control import PhysicalShot5D, WPA_NINE_BALL_BREAK_DECODER


@dataclass(frozen=True)
class PhysicalErrorModel:
    """Standard deviations in FastFiz physical units, NOT the paper's units."""

    azimuth_deg: float = 0.25
    speed_m_s: float = 0.05
    elevation_deg: float = 0.3
    tip_x_mm: float = 0.35
    tip_y_mm: float = 0.35

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in vars(self).values()):
            raise ValueError("Shot-noise standard deviations must be finite and nonnegative")

    def perturb(self, shot, rng):
        noise = rng.normal(0.0, [
            self.speed_m_s, self.azimuth_deg, self.elevation_deg,
            self.tip_x_mm, self.tip_y_mm,
        ])
        x, y = shot.tip_x_mm + float(noise[3]), shot.tip_y_mm + float(noise[4])
        norm = math.hypot(x, y)
        if norm > 11.99:
            x, y = (x * 11.99 / norm, y * 11.99 / norm)
        return PhysicalShot5D(
            speed_m_s=max(0.05, shot.speed_m_s + float(noise[0])),
            azimuth_deg=shot.azimuth_deg + float(noise[1]),
            elevation_deg=float(np.clip(shot.elevation_deg + noise[2], 0.0, 69.0)),
            tip_x_mm=x, tip_y_mm=y,
        )


@dataclass(frozen=True)
class CandidateScore:
    goal: NineBallGoal
    shot: PhysicalShot5D
    value: float
    pot_rate: float
    accepted: int
    invalid: int


def _rack_fingerprint(state, context, seed):
    encoded = encode_rack_v3(state, context)
    digest = hashlib.sha256(encoded.tobytes() + str(seed).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little")


class PickPocketAdaptedAgent(GeometryNineBallAgent):
    """Root-level physical Monte Carlo baseline with independent shot generation.

    This adapter uses the project physics/rules for every candidate and keeps
    the learned 5D Micro out of regular shot selection. Its shared break actor
    is a deliberate control so break policy does not dominate comparisons.
    """

    def __init__(
        self, break_micro_checkpoint, *, device="cpu", samples=3,
        max_candidates=8, seed=32, error_model=None,
        safety_threshold=0.65,
    ):
        super().__init__()
        if samples < 1 or max_candidates < 1 or seed < 0:
            raise ValueError("samples/max_candidates must be positive, seed >= 0")
        if not 0 <= safety_threshold <= 1:
            raise ValueError("safety_threshold must lie in [0,1]")
        self.samples = int(samples)
        self.max_candidates = int(max_candidates)
        self.seed = int(seed)
        self.error_model = error_model or PhysicalErrorModel()
        self.safety_threshold = float(safety_threshold)
        self.break_agent = NineBallMicroSACAgent(
            break_micro_checkpoint, device=device, deterministic=True,
            learned_break=True,
        )
        self.break_agent.break_decoder = WPA_NINE_BALL_BREAK_DECODER
        self.rules = NineBallRules(wpa_break_rules=True)
        self.simulator = None
        self.last_diagnostics = None

    def _ensure_simulator(self, state):
        if self.simulator is None:
            self.simulator = MultiBallFastFizSimulator(state.config)
            self.simulator.require_head_string_crossings()

    def _nominal(self, state, target, pocket_id, speed, tip_y=0.0):
        goal = NineBallGoal("DIRECT_ATTACK", target, pocket_id)
        angle = reference_angle_deg(state, goal)
        return goal, PhysicalShot5D(speed, angle, 10.0, 0.0, tip_y)

    def _attack_candidates(self, state):
        target = min(state.on_table_object_balls())
        options = []
        for pocket in direct_pocket_candidates(state, target):
            for speed in (1.2, 1.9, 2.7):
                for tip_y in (-3.0, 0.0, 3.0):
                    options.append(self._nominal(state, target, pocket, speed, tip_y))
        # Search breadth is deliberately bounded, but include different pockets.
        pockets = sorted({goal.target_pocket for goal, _ in options})
        selected = []
        for pocket in pockets:
            shots = [(goal, shot) for goal, shot in options if goal.target_pocket == pocket]
            selected.append(shots[len(shots) // 2])
        remaining = [item for item in options if item not in selected]
        while len(selected) < self.max_candidates and remaining:
            selected.append(remaining.pop(len(remaining) // 2))
        return selected[:self.max_candidates]

    def _safety_candidates(self, state):
        target = min(state.on_table_object_balls())
        ref = reference_angle_deg(state, NineBallGoal("DIRECT_ATTACK", target))
        return [
            (
                NineBallGoal("SAFETY", target),
                PhysicalShot5D(speed, ref + offset, 10.0, 0.0, 0.0),
            )
            for offset in (-18.0, 0.0, 18.0)
            for speed in (0.75, 1.25)
        ]

    @staticmethod
    def _value(state, goal, physical, outcome):
        player = state.current_player
        if outcome.rack_winner is not None:
            return 2.0 if outcome.rack_winner == player else -2.0, False
        if outcome.foul:
            return -1.0, False
        if getattr(outcome, "illegal_break", False):
            return -0.5, False
        if goal.kind == "DIRECT_ATTACK":
            ball_index = state.array_index_for_ball(goal.target_ball)
            success = (
                physical.pocket_indices[ball_index] == goal.target_pocket
                and not outcome.turn_changed
            )
        else:
            success = False
        next_state = outcome.next_state
        other_targets = next_state.on_table_object_balls()
        mobility = 0.0
        if other_targets:
            next_target = min(other_targets)
            mobility = min(6, len(tuple(direct_pocket_candidates(next_state, next_target)))) / 6.0
        if success:
            return 0.7 + 0.4 * mobility, True
        if outcome.turn_changed:
            # An opponent with poor follow-up options is preferable.
            return -0.2 - 0.5 * mobility, False
        return -0.3, False

    def _score_candidate(self, state, context, goal, shot, *, rng):
        values = []
        pots = 0
        invalid = 0
        for _ in range(self.samples):
            noisy = self.error_model.perturb(shot, rng)
            try:
                physical, outcome = simulate_nine_ball_shot(
                    self.simulator, self.rules, state, context,
                    target_ball=goal.target_ball, shot=noisy,
                )
            except InvalidPhysicalTransition:
                invalid += 1
                # Not an actual foul or opponent victory; penalize
                # non-executability only in candidate selection.
                continue
            value, pot = self._value(state, goal, physical, outcome)
            values.append(value)
            pots += int(pot)
        if not values:
            return CandidateScore(goal, shot, float("-inf"), 0.0, 0, invalid)
        # Invalid samples receive zero utility; no invented legal outcome.
        return CandidateScore(
            goal, shot, sum(values) / self.samples,
            pots / self.samples, len(values), invalid,
        )

    def _select(self, state, context):
        self._ensure_simulator(state)
        base_seed = _rack_fingerprint(state, context, self.seed)
        attack_scores = []
        attempts = 0
        for candidate_index, (goal, shot) in enumerate(self._attack_candidates(state)):
            attempts += 1
            # Try same-aim cue clearance for the *nominal* shot, as in
            # the proposed search. No changes to target/pocket.
            try:
                shot, _, _, _ = simulate_with_cue_clearance(
                    lambda attempt: simulate_nine_ball_shot(
                        self.simulator, self.rules, state, context,
                        target_ball=goal.target_ball, shot=attempt,
                    ),
                    shot,
                )
            except InvalidPhysicalTransition:
                continue
            rng = np.random.default_rng(base_seed + candidate_index)
            attack_scores.append(self._score_candidate(state, context, goal, shot, rng=rng))

        best_attack = max(attack_scores, key=lambda x: x.value, default=None)
        best = best_attack
        safety_evaluated = False
        if best_attack is None or best_attack.pot_rate < self.safety_threshold:
            safety_evaluated = True
            for index, (goal, shot) in enumerate(self._safety_candidates(state)):
                try:
                    shot, _, _, _ = simulate_with_cue_clearance(
                        lambda attempt: simulate_nine_ball_shot(
                            self.simulator, self.rules, state, context,
                            target_ball=goal.target_ball, shot=attempt,
                        ),
                        shot,
                    )
                except InvalidPhysicalTransition:
                    continue
                rng = np.random.default_rng(base_seed + 10000 + index)
                score = self._score_candidate(state, context, goal, shot, rng=rng)
                if best is None or score.value > best.value:
                    best = score

        self.last_diagnostics = {
            "candidate_count": attempts,
            "valid_candidates": len(attack_scores),
            "safety_evaluated": safety_evaluated,
            "samples_per_candidate": self.samples,
            "selected_value": best.value if best is not None else None,
            "selected_pot_rate": best.pot_rate if best is not None else None,
        }
        if best is None or best.value == float("-inf"):
            # Emergency fixed lowest-ball aim, never a fabricated outcome.
            target = min(state.on_table_object_balls())
            goal = NineBallGoal("DIRECT_ATTACK", target)
            return ShotPlan(
                target, PhysicalShot5D(1.6, reference_angle_deg(state, goal), 10.0, 0.0, 0.0),
                "PICKPOCKET_V0_UNRESOLVED_FALLBACK", macro_goal=goal,
            )
        selected_goal = best.goal
        if selected_goal.kind == "SAFETY":
            selected_goal = replace(selected_goal, explicit_shot=best.shot)
        return ShotPlan(
            best.goal.target_ball, best.shot, "PICKPOCKET_V0_MC_" + best.goal.kind,
            target_pocket=best.goal.target_pocket, macro_goal=selected_goal,
        )

    def choose_shot(self, state, context=None):
        return self.choose_match_shot(state, context, None)

    def choose_match_shot(self, state, context, match_position):
        if state.ball_in_hand:
            raise ValueError("Cue-ball placement required before selecting a shot")
        context = context or NineBallContext()
        target = min(state.on_table_object_balls())
        if context.pending_choice is not None:
            goal = NineBallGoal("ACCEPT_TABLE", target)
            return ShotPlan(target, None, "PICKPOCKET_V0_ACCEPT_TABLE", macro_goal=goal)
        if state.break_shot:
            goal = NineBallGoal("BREAK", target)
            plan = self.break_agent.plan_goal(state, goal, context)
            self._ensure_simulator(state)
            try:
                corrected, _, _, _ = simulate_with_cue_clearance(
                    lambda attempt: simulate_nine_ball_shot(
                        self.simulator, self.rules, state, context,
                        target_ball=target, shot=attempt,
                    ),
                    plan.shot,
                )
                plan = replace(plan, shot=corrected)
            except InvalidPhysicalTransition:
                # Keep the original physical request; the live collector
                # will mark a non-executable shot as unfinished, not foul.
                pass
            return replace(plan, goal_label="PICKPOCKET_V0_SHARED_BREAK_MICRO")
        return self._select(state, context)
