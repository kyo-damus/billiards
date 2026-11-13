"""9-ball version-3 physical PUCT with turn-aware terminal match values.

Unlike the older two-object-ball MCTS, this tree uses RackState/NineBallContext,
NineBallGoal, the 5D Micro, MultiBallFastFiz, and NineBallRules. Candidate
Policy supplies priors; optional State V estimates discounted match outcomes
at depth-limited leaves. The selected-action Macro Q is intentionally NOT
used to score unexecuted candidates. This is a bounded deterministic planner,
not a learned dynamics model or proof of game-strength improvement.
"""

from dataclasses import dataclass, field, replace
import hashlib
import math

import numpy as np
import torch

from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.nine_ball_encoding_v3 import (
    encode_goal_v3, encode_macro_state, generate_nine_ball_goals,
)
from src.game.nine_ball_rules import NineBallContext, NineBallRules
from src.game.nine_ball_session import NineBallRackSession, create_nine_ball_rack
from src.game.match import MatchPosition
from src.game.nine_ball_physics_guard import (
    InvalidPhysicalTransition, NativeShotRejected, simulate_nine_ball_shot, validate_rack_layout,
)
from src.game.nine_ball_position import generate_outcome_position_goals
from src.game.rack_state import TableGameConfig


CUE_STICK_COLLISION = 128  # FastFiz TableState precondition flag.


@dataclass
class SearchEdge:
    goal: object
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    child: object = None
    terminal_reward: float = 0.0
    invalid_reason: str | None = None
    plan: object | None = None
    cue_clearance_trials: int = 0

    @property
    def q(self):
        """Empirical action value in the PARENT player's perspective."""
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class SearchNode:
    state: object
    context: NineBallContext
    winner: int | None = None
    match_position: MatchPosition | None = None
    visits: int = 0
    expanded: bool = False
    edges: list[SearchEdge] = field(default_factory=list)


class NineBallMCTS:
    """Deterministic candidate PUCT, max_depth measured in physical shots."""

    def __init__(
        self, micro_agent, *,
        policy_network=None,
        value_network=None,
        simulator=None,
        rules=None,
        simulations=24,
        max_depth=2,
        c_puct=1.5,
        gamma=0.99,
        root_min_visits=1,
        learned_safety=False,
        learned_break=False,
        outcome_position=False,
        match_aware=False,
        rack_value=False,
    ):
        if simulations < 1 or max_depth < 1 or root_min_visits < 0:
            raise ValueError("simulations and max_depth must be positive")
        if c_puct < 0 or not 0 < gamma <= 1:
            raise ValueError("invalid PUCT/gamma parameters")
        self.micro_agent = micro_agent
        self.policy = policy_network
        self.value = value_network
        self.device = micro_agent.device
        self.simulator = simulator or MultiBallFastFizSimulator(
            TableGameConfig.nine_ball_inventory()
        )
        self.rules = rules or NineBallRules()
        self.simulations = simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.gamma = gamma
        self.root_min_visits = root_min_visits
        self.learned_safety = learned_safety
        self.learned_break = learned_break
        self.outcome_position = outcome_position
        self.match_aware = match_aware
        self.rack_value = rack_value
        self.last_root = None
        self.last_report = None
        self.last_plan = None

    def _prepare(self, node):
        """Reproduce live collector's deterministic ball-in-hand placement."""
        if node.winner is not None or not node.state.ball_in_hand:
            return
        session = NineBallRackSession(
            self.simulator, node.state, rules=self.rules,
        )
        for x, y in self.micro_agent.cue_placement_candidates(node.state):
            try:
                node.state = session.place_cue_ball(x, y)
                return
            except ValueError as exc:
                if "overlaps" not in str(exc) and "outside playing surface" not in str(exc):
                    raise
        raise RuntimeError("No valid cue-ball placement found in MCTS")

    def _expand(self, node):
        if node.expanded or node.winner is not None:
            return
        validate_rack_layout(node.state)
        self._prepare(node)
        validate_rack_layout(node.state)
        if self.outcome_position and not node.state.break_shot:
            candidates = generate_outcome_position_goals(
                node.state, node.context, self.simulator, self.rules,
                learned_safety=self.learned_safety,
            )
        else:
            candidates = generate_nine_ball_goals(
                node.state, learned_safety=self.learned_safety, learned_break=self.learned_break,
            )
        if not candidates:
            raise RuntimeError("Nonterminal rack has no nine-ball candidate")
        if self.policy is None:
            priors = np.full(len(candidates), 1.0 / len(candidates))
        else:
            s = torch.as_tensor(
                encode_macro_state(node.state, node.context, node.match_position),
                dtype=torch.float32, device=self.device,
            )
            c = torch.as_tensor(
                np.stack([encode_goal_v3(goal) for goal in candidates]),
                dtype=torch.float32, device=self.device,
            )
            with torch.no_grad():
                logits = self.policy(s, c).reshape(-1)
                probabilities = torch.softmax(logits, dim=0)
            priors = probabilities.detach().cpu().numpy()
        node.edges = [
            SearchEdge(goal=goal, prior=float(prior))
            for goal, prior in zip(candidates, priors)
        ]
        node.expanded = True

    def _leaf_value(self, node):
        """Estimate value from the perspective of the player to act."""
        if node.winner is not None:
            return 0.0  # Rack-boundary values live on the parent edge.
        # Expand candidates only if a subsequent simulation traverses this leaf.
        self._prepare(node)
        validate_rack_layout(node.state)
        position = node.match_position
        value = 0.0
        if self.value is not None:
            encoded = torch.as_tensor(
                encode_macro_state(node.state, node.context, None if self.rack_value else position),
                dtype=torch.float32, device=self.device,
            )
            with torch.no_grad():
                prediction = self.value(encoded.unsqueeze(0))
            value = float(np.clip(float(prediction.reshape(-1)[0].item()), -1.0, 1.0))
        if position is not None and (self.value is None or self.rack_value):
            player = node.state.current_player
            win = position.after_rack(player).fair_match_value(player)
            loss = position.after_rack(1 - player).fair_match_value(player)
            return (win + loss + value * (win - loss)) * 0.5
        return value

    def _select_edge(self, node, *, root=False):
        # A failed native transition is not a low-reward legal candidate.
        eligible = [e for e in node.edges if e.invalid_reason is None]
        if not eligible:
            return None
        if root and self.root_min_visits:
            least = min(edge.visits for edge in eligible)
            if least < self.root_min_visits:
                return max(
                    (e for e in eligible if e.visits == least),
                    key=lambda e: e.prior,
                )
        scale = math.sqrt(max(1, node.visits))
        return max(
            eligible,
            key=lambda e: (
                e.q + self.c_puct * e.prior * scale / (1 + e.visits),
                e.prior,
            ),
        )

    def _simulate_plan(self, node, plan):
        _, outcome = simulate_nine_ball_shot(
            self.simulator, self.rules, node.state, node.context,
            target_ball=plan.target_ball, shot=plan.shot,
        )
        return outcome

    def _resolve_cue_clearance(self, node, edge, plan, rejected):
        """Retry the same aiming line from the same board with a raised cue."""
        shot = plan.shot
        elevations = (20.0, 30.0, 40.0, 50.0, 60.0, 69.0)
        # First retain the original power and spin. Then test a higher
        # contact point and sufficient speed for the more elevated stroke.
        variants = [replace(shot, elevation_deg=e) for e in elevations if e > shot.elevation_deg]
        variants.extend(
            replace(
                shot, elevation_deg=e, speed_m_s=max(shot.speed_m_s, 2.2),
                tip_x_mm=0.0, tip_y_mm=8.0,
            )
            for e in elevations if e >= max(30.0, shot.elevation_deg)
        )
        first_foul = None
        for variant in variants:
            edge.cue_clearance_trials += 1
            retry = replace(plan, shot=variant)
            try:
                outcome = self._simulate_plan(node, retry)
            except InvalidPhysicalTransition:
                continue
            if not outcome.foul:
                return retry, outcome
            if first_foul is None:
                first_foul = (retry, outcome)
        if first_foul is not None:
            return first_foul
        raise InvalidPhysicalTransition(
            f"{rejected}; cue-clearance trials exhausted ({edge.cue_clearance_trials})"
        ) from rejected

    def _step(self, node, edge):
        """Return False only for a diagnosed physically invalid shot."""
        try:
            plan = self.micro_agent.plan_goal(node.state, edge.goal, node.context)
            try:
                outcome = self._simulate_plan(node, plan)
            except NativeShotRejected as exc:
                if exc.precondition != CUE_STICK_COLLISION or exc.placement != 0:
                    raise
                plan, outcome = self._resolve_cue_clearance(node, edge, plan, exc)
        except InvalidPhysicalTransition as exc:
            edge.invalid_reason = str(exc)
            return False
        edge.plan = plan
        edge.child = SearchNode(
            state=outcome.next_state,
            context=outcome.next_context,
            winner=outcome.rack_winner,
            match_position=node.match_position,
        )
        if edge.child.winner is not None:
            player = node.state.current_player
            if node.match_position is None:
                edge.terminal_reward = 1.0 if edge.child.winner == player else -1.0
            else:
                after = node.match_position.after_rack(edge.child.winner)
                if after.winner is not None:
                    edge.terminal_reward = after.fair_match_value(player)
                elif self.value is None or self.rack_value:
                    edge.terminal_reward = self.gamma * after.fair_match_value(player)
                else:
                    # The future rack order is unknown to both players. Use an
                    # independent layout shared across branches of this search.
                    next_rack = create_nine_ball_rack(breaker=after.breaker, seed=self._bootstrap_seed)
                    continuation = self._leaf_value(SearchNode(
                        next_rack, NineBallContext(), match_position=after,
                    ))
                    sign = 1.0 if after.breaker == player else -1.0
                    edge.terminal_reward = self.gamma * sign * continuation
        return True

    @staticmethod
    def _discard_backups(accepted_paths, invalid_edge):
        """Remove previous leaf evaluations whose path used an invalidated edge."""
        if not invalid_edge.visits:
            return
        retained = []
        for backup in accepted_paths:
            if not any(edge is invalid_edge for _, edge, _ in backup):
                retained.append(backup)
                continue
            for parent, edge, value in backup:
                edge.visits -= 1
                edge.value_sum = edge.value_sum - value if edge.visits else 0.0
                parent.visits -= 1
        accepted_paths[:] = retained

    def _simulate(self, root, accepted_paths):
        node = root
        path = []
        leaf = 0.0
        for depth in range(self.max_depth):
            self._expand(node)
            edge = self._select_edge(node, root=(node is root))
            if edge is None:
                if not path:
                    raise InvalidPhysicalTransition(
                        "No executable nine-ball root MCTS candidate: " +
                        "; ".join(
                            f"{i}: {e.invalid_reason}"
                            for i, e in enumerate(root.edges)
                        )
                    )
                # Discard a simulated path whose continuation has no
                # executable native action; never invent a leaf reward.
                invalid_edge = path[-1][1]
                invalid_edge.invalid_reason = "no executable descendant"
                self._discard_backups(accepted_paths, invalid_edge)
                return False
            fresh = edge.child is None
            if fresh and not self._step(node, edge):
                return False
            path.append((node, edge))
            node = edge.child
            if node.winner is not None:
                leaf = 0.0
                break
            if fresh or depth + 1 >= self.max_depth:
                leaf = self._leaf_value(node)
                break
        else:
            leaf = self._leaf_value(node)

        # Each backed-up value belongs to that edge's ACTING player.
        # Rack-boundary values are already in the parent perspective.
        # Within a rack, child value is discounted
        # and sign-flipped when the next player differs.
        value = leaf
        backup = []
        for parent, edge in reversed(path):
            child = edge.child
            if child.winner is not None:
                value = edge.terminal_reward
            else:
                sign = (
                    1.0
                    if parent.state.current_player == child.state.current_player
                    else -1.0
                )
                value = self.gamma * sign * value
            edge.visits += 1
            edge.value_sum += value
            parent.visits += 1
            backup.append((parent, edge, value))
        accepted_paths.append(backup)
        return True

    def search(self, state, context=None, match_position=None):
        self.last_plan = None
        self.last_report = None
        self.last_root = None
        if state.ball_in_hand:
            raise ValueError("Live match must place cue ball before searching")
        context = NineBallContext() if context is None else context
        if self.match_aware and (match_position is None or match_position.winner is not None):
            raise ValueError("Match-aware MCTS requires an unfinished match position")
        if self.match_aware and self.value is not None and not self.rack_value:
            digest = hashlib.blake2b(state.ball_positions.tobytes(), digest_size=8).digest()
            self._bootstrap_seed = int.from_bytes(digest, "little")
        root = SearchNode(
            state=state, context=context, match_position=match_position if self.match_aware else None,
        )
        self._expand(root)
        accepted_paths = []
        attempts, rejected = 0, 0
        # Each rejected attempt invalidates an edge; future attempts
        # then explore other paths. Also cap unexpected loops explicitly.
        max_attempts = self.simulations + max(
            128, 8 * self.simulations * self.max_depth,
        )
        while len(accepted_paths) < self.simulations:
            if attempts >= max_attempts:
                raise InvalidPhysicalTransition(
                    f"MCTS invalid-transition limit reached: "
                    f"valid={len(accepted_paths)}, rejected={rejected}"
                )
            attempts += 1
            if not self._simulate(root, accepted_paths):
                rejected += 1
        self.last_root = root
        valid_edges = [e for e in root.edges if e.invalid_reason is None]
        if not valid_edges:
            raise InvalidPhysicalTransition("No executable root candidate")
        # Resolve visit ties by observed parent-perspective Q, then prior.
        chosen = max(valid_edges, key=lambda e: (e.visits, e.q, e.prior))
        self.last_plan = chosen.plan
        self.last_report = {
            "simulations": len(accepted_paths),
            "requested_simulations": self.simulations,
            "rejected_invalid_simulations": rejected,
            "discarded_invalidated_simulations": attempts - rejected - len(accepted_paths),
            "root_invalid_reasons": [
                e.invalid_reason for e in root.edges
            ],
            "root_cue_clearance_trials": [e.cue_clearance_trials for e in root.edges],
            "chosen_cue_clearance_adjusted": chosen.cue_clearance_trials > 0,
            "max_depth": self.max_depth,
            "uses_learned_state_v": self.value is not None,
            "uses_selected_action_q": False,
            "root_candidates": len(root.edges),
            "chosen_index": next(
                i for i, edge in enumerate(root.edges) if edge is chosen
            ),
            "root_visits": [edge.visits for edge in root.edges],
            "root_priors": [edge.prior for edge in root.edges],
            "root_empirical_q": [edge.q for edge in root.edges],
            "goal_v3": [encode_goal_v3(edge.goal).tolist() for edge in root.edges],
        }
        return chosen.goal
