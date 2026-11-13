"""9-ball version-3 physical PUCT with turn-aware terminal match values.

Unlike the older two-object-ball MCTS, this tree uses RackState/NineBallContext,
NineBallGoal, the 5D Micro, MultiBallFastFiz, and NineBallRules. Candidate
Policy supplies priors; optional State V estimates discounted match outcomes
at depth-limited leaves. The selected-action Macro Q is intentionally NOT
used to score unexecuted candidates. This is a bounded deterministic planner,
not a learned dynamics model or proof of game-strength improvement.
"""

from dataclasses import dataclass, field
import math

import numpy as np
import torch

from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.nine_ball_encoding_v3 import (
    encode_goal_v3, encode_rack_v3, generate_nine_ball_goals,
)
from src.game.nine_ball_rules import NineBallContext, NineBallRules
from src.game.nine_ball_session import NineBallRackSession
from src.game.nine_ball_physics_guard import (
    InvalidPhysicalTransition, validate_physical_step, validate_rack_layout,
)
from src.game.rack_state import TableGameConfig


@dataclass
class SearchEdge:
    goal: object
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    child: object = None
    terminal_reward: float = 0.0
    invalid_reason: str | None = None

    @property
    def q(self):
        """Empirical action value in the PARENT player's perspective."""
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class SearchNode:
    state: object
    context: NineBallContext
    winner: int | None = None
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
        self.last_root = None
        self.last_report = None

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
        candidates = generate_nine_ball_goals(node.state, learned_safety=self.learned_safety)
        if not candidates:
            raise RuntimeError("Nonterminal rack has no nine-ball candidate")
        if self.policy is None:
            priors = np.full(len(candidates), 1.0 / len(candidates))
        else:
            s = torch.as_tensor(
                encode_rack_v3(node.state, node.context),
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
        """Perspective: player to act in node.state; rewards are +/-1."""
        if node.winner is not None:
            return 0.0  # terminal rewards live on the parent edge
        self._expand(node)
        if self.value is None:
            return 0.0
        encoded = torch.as_tensor(
            encode_rack_v3(node.state, node.context),
            dtype=torch.float32, device=self.device,
        )
        with torch.no_grad():
            prediction = self.value(encoded.unsqueeze(0))
        # MC return targets are bounded by [-1,1].
        return float(np.clip(float(prediction.reshape(-1)[0].item()), -1.0, 1.0))

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

    def _step(self, node, edge):
        """Return False only for a diagnosed physically invalid shot."""
        try:
            validate_rack_layout(node.state)
            plan = self.micro_agent.plan_goal(node.state, edge.goal, node.context)
            self.simulator.restore(node.state)
            physical = self.simulator.step(
                target_ball=plan.target_ball, shot=plan.shot,
            )
            validate_physical_step(physical)
            outcome = self.rules.apply(node.state, physical, node.context)
            validate_rack_layout(outcome.next_state)
        except InvalidPhysicalTransition as exc:
            edge.invalid_reason = str(exc)
            return False
        except RuntimeError as exc:
            if "Bad Shot Exception" not in str(exc):
                raise
            edge.invalid_reason = f"FastFiz Bad Shot Exception: {exc}"
            return False
        edge.child = SearchNode(
            state=outcome.next_state,
            context=outcome.next_context,
            winner=outcome.rack_winner,
        )
        if edge.child.winner is not None:
            edge.terminal_reward = (
                1.0 if edge.child.winner == node.state.current_player else -1.0
            )
        return True

    def _simulate(self, root):
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
                path[-1][1].invalid_reason = "no executable descendant"
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
        # Terminal +/-1 reward is immediate; child value is discounted
        # and sign-flipped when the next player differs.
        value = leaf
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
        return True

    def search(self, state, context=None):
        if state.ball_in_hand:
            raise ValueError("Live match must place cue ball before searching")
        context = NineBallContext() if context is None else context
        root = SearchNode(state=state, context=context)
        self._expand(root)
        successful, rejected = 0, 0
        # Each rejected attempt invalidates an edge; future attempts
        # then explore other paths. Also cap unexpected loops explicitly.
        max_attempts = self.simulations + max(
            128, 8 * self.simulations * self.max_depth,
        )
        while successful < self.simulations:
            if successful + rejected >= max_attempts:
                raise InvalidPhysicalTransition(
                    f"MCTS invalid-transition limit reached: "
                    f"valid={successful}, rejected={rejected}"
                )
            if self._simulate(root):
                successful += 1
            else:
                rejected += 1
        self.last_root = root
        valid_edges = [e for e in root.edges if e.invalid_reason is None]
        if not valid_edges:
            raise InvalidPhysicalTransition("No executable root candidate")
        # Resolve visit ties by observed parent-perspective Q, then prior.
        chosen = max(valid_edges, key=lambda e: (e.visits, e.q, e.prior))
        self.last_report = {
            "simulations": successful,
            "requested_simulations": self.simulations,
            "rejected_invalid_simulations": rejected,
            "root_invalid_reasons": [
                e.invalid_reason for e in root.edges
            ],
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
