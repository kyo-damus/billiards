"""Small deterministic physics-tree tests for 9-ball PUCT semantics."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import src.game.nine_ball_mcts_v3 as module
from src.game.nine_ball_agents import GeometryNineBallAgent, ShotPlan
from src.game.nine_ball_encoding_v3 import NineBallGoal
from src.game.nine_ball_mcts_v3 import NineBallMCTS, SearchNode
from src.game.nine_ball_rules import NineBallContext
from src.game.nine_ball_session import create_nine_ball_rack


class FakeMicro:
    device = torch.device("cpu")

    def __init__(self):
        self.goal_calls = []

    def plan_goal(self, state, goal, context):
        self.goal_calls.append((state, goal, context))
        return SimpleNamespace(target_ball=1, shot=goal.target_pocket)

    def cue_placement_candidates(self, state):
        return GeometryNineBallAgent().cue_placement_candidates(state)


class FakePhysics:
    def __init__(self):
        self.restored = []
        self.state = None

    def restore(self, state):
        self.state = state
        self.restored.append(state)

    def step(self, *, target_ball, shot):
        assert self.state is not None  # each counterfactual is restored
        return SimpleNamespace(origin=self.state, shot=shot)


def test_root_puct_prefers_observed_legal_terminal_win(monkeypatch):
    direct = NineBallGoal("DIRECT_ATTACK", 1, 0)
    alternative = NineBallGoal("DIRECT_ATTACK", 1, 1)
    root_state = SimpleNamespace(current_player=0, ball_in_hand=False)

    monkeypatch.setattr(
        module, "generate_nine_ball_goals",
        lambda state: (direct, alternative),
    )

    class Rules:
        def apply(self, before, physical, context):
            assert physical.origin is root_state
            return SimpleNamespace(
                next_state=SimpleNamespace(
                    current_player=0, ball_in_hand=False,
                ),
                next_context=context,
                rack_winner=0 if physical.shot == 0 else 1,
            )

    physics = FakePhysics()
    micro = FakeMicro()
    search = NineBallMCTS(
        micro, simulator=physics, rules=Rules(),
        simulations=2, max_depth=1, root_min_visits=1,
    )
    choice = search.search(root_state, NineBallContext())
    assert choice == direct
    assert search.last_report["root_visits"] == [1, 1]
    assert search.last_report["root_empirical_q"] == [1.0, -1.0]
    assert physics.restored == [root_state, root_state]
    assert len(micro.goal_calls) == 2


def test_opponent_win_is_negative_to_root_even_if_its_own_value_is_positive(
    monkeypatch,
):
    first = NineBallGoal("DIRECT_ATTACK", 1, 0)
    reply = NineBallGoal("DIRECT_ATTACK", 1, 1)
    root_state = SimpleNamespace(current_player=0, ball_in_hand=False)
    opponent = SimpleNamespace(current_player=1, ball_in_hand=False)

    monkeypatch.setattr(
        module, "generate_nine_ball_goals",
        lambda state: (first,) if state is root_state else (reply,),
    )

    class Rules:
        def apply(self, before, physical, context):
            return SimpleNamespace(
                next_state=opponent,
                next_context=context,
                rack_winner=1 if physical.shot == 1 else None,
            )

    search = NineBallMCTS(
        FakeMicro(), simulator=FakePhysics(), rules=Rules(),
        simulations=3, max_depth=2, gamma=0.5,
    )
    choice = search.search(root_state, NineBallContext())
    root_edge = search.last_root.edges[0]
    reply_edge = root_edge.child.edges[0]
    assert choice == first
    assert reply_edge.q == pytest.approx(1.0)
    # First root visit has zero leaf V; next visits see opponent terminal win.
    assert root_edge.q < 0
    assert search.last_report["uses_selected_action_q"] is False


def test_ball_in_hand_node_uses_game_session_placement():
    original = create_nine_ball_rack(breaker=1, seed=10)
    scratched = replace(original, ball_in_hand=True)
    physics = FakePhysics()
    search = NineBallMCTS(
        FakeMicro(), simulator=physics,
        simulations=1, max_depth=1,
    )
    node = SearchNode(scratched, NineBallContext((1, 0)))
    search._prepare(node)
    assert not node.state.ball_in_hand
    assert node.state.current_player == scratched.current_player
    assert node.context.consecutive_fouls == (1, 0)
    assert physics.restored
    # Root is never mutated, and the pre-shot cue location was relocated.
    assert scratched.ball_in_hand


def test_invalid_search_parameters():
    with pytest.raises(ValueError):
        NineBallMCTS(FakeMicro(), simulations=0)
    with pytest.raises(ValueError):
        NineBallMCTS(FakeMicro(), simulations=1, gamma=0)
