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
from src.game.match import MatchPosition
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
        lambda state, **kwargs: (direct, alternative),
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
        lambda state, **kwargs: (first,) if state is root_state else (reply,),
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


def test_nineball_mcts_agent_forwards_search_discount(monkeypatch):
    from src.game.nine_ball_mcts_agent import NineBallMCTSAgent
    import src.game.nine_ball_mcts_agent as agent_module

    class DummyMicro:
        def __init__(self, *args, **kwargs):
            self.macro_policy = None

    class DummyMcts:
        def __init__(self, micro, **kwargs):
            self.gamma = kwargs["gamma"]

    monkeypatch.setattr(
        agent_module.NineBallMicroSACAgent, "__init__", DummyMicro.__init__
    )
    monkeypatch.setattr(agent_module, "NineBallMCTS", DummyMcts)
    candidate = NineBallMCTSAgent("unused", gamma=0.9)
    assert candidate.searcher.gamma == pytest.approx(0.9)


def test_mcts_rejects_overlapping_next_state_without_fabricating_visits(monkeypatch):
    from dataclasses import replace
    from src.game.nine_ball_physics_guard import InvalidPhysicalTransition

    rack = replace(create_nine_ball_rack(breaker=0, seed=11), break_shot=False)
    bad_positions = rack.ball_positions.copy()
    bad_positions[2] = bad_positions[5]
    overlapping = replace(rack, ball_positions=bad_positions)

    a = NineBallGoal("DIRECT_ATTACK", 1, 0)
    b = NineBallGoal("DIRECT_ATTACK", 1, 1)
    monkeypatch.setattr(module, "generate_nine_ball_goals", lambda state, **kwargs: (a, b))

    class Rules:
        def apply(self, before, physical, context):
            if physical.shot == 0:
                return SimpleNamespace(
                    next_state=overlapping, next_context=context, rack_winner=None,
                )
            return SimpleNamespace(
                next_state=rack, next_context=context, rack_winner=0,
            )

    search = NineBallMCTS(
        FakeMicro(), simulator=FakePhysics(), rules=Rules(),
        simulations=3, max_depth=1, root_min_visits=1,
    )
    assert search.search(rack) == b
    assert search.last_report["root_visits"] == [0, 3]
    assert search.last_report["rejected_invalid_simulations"] == 1
    assert "overlapping balls" in search.last_report["root_invalid_reasons"][0]
    assert sum(search.last_report["root_visits"]) == 3


def test_native_bad_shot_is_excluded_unrelated_runtime_error_is_not(monkeypatch):
    from src.game.nine_ball_physics_guard import InvalidPhysicalTransition

    a = NineBallGoal("DIRECT_ATTACK", 1, 0)
    b = NineBallGoal("DIRECT_ATTACK", 1, 1)
    root = SimpleNamespace(current_player=0, ball_in_hand=False)
    monkeypatch.setattr(module, "generate_nine_ball_goals", lambda state, **kwargs: (a, b))

    class Physics(FakePhysics):
        def step(self, *, target_ball, shot):
            if shot == 0:
                raise RuntimeError("Bad Shot Exception")
            return super().step(target_ball=target_ball, shot=shot)

    class Rules:
        def apply(self, before, physical, context):
            return SimpleNamespace(
                next_state=root, next_context=context, rack_winner=0,
            )

    search = NineBallMCTS(
        FakeMicro(), simulator=Physics(), rules=Rules(),
        simulations=2, max_depth=1, root_min_visits=1,
    )
    assert search.search(root) == b
    assert search.last_report["root_visits"] == [0, 2]
    assert search.last_report["rejected_invalid_simulations"] == 1

    class OtherPhysics(FakePhysics):
        def step(self, *, target_ball, shot):
            raise RuntimeError("Unrelated native corruption")

    search = NineBallMCTS(
        FakeMicro(), simulator=OtherPhysics(), rules=Rules(),
        simulations=1, max_depth=1,
    )
    with pytest.raises(RuntimeError, match="Unrelated"):
        search.search(root)

    class AlwaysBad(FakePhysics):
        def step(self, *, target_ball, shot):
            raise RuntimeError("Bad Shot Exception")

    search = NineBallMCTS(
        FakeMicro(), simulator=AlwaysBad(), rules=Rules(),
        simulations=1, max_depth=1,
    )
    with pytest.raises(InvalidPhysicalTransition, match="No executable"):
        search.search(root)


@pytest.mark.parametrize("depth", (2, 3))
def test_invalidated_descendant_rolls_back_preliminary_root_visits(monkeypatch, depth):
    first, alternative, reply, last = (
        NineBallGoal("DIRECT_ATTACK", 1, pocket) for pocket in range(4)
    )
    states = [SimpleNamespace(index=i, current_player=0, ball_in_hand=False) for i in range(3)]
    goals = [(first, alternative), (reply,), (last,)]
    monkeypatch.setattr(module, "generate_nine_ball_goals", lambda state, **kw: goals[state.index])

    class Physics(FakePhysics):
        def step(self, *, target_ball, shot):
            if shot == (2 if depth == 2 else 3):
                raise RuntimeError("Bad Shot Exception")
            return super().step(target_ball=target_ball, shot=shot)

    class Rules:
        def apply(self, before, physical, context):
            return SimpleNamespace(
                next_state=states[min(before.index + 1, 2)],
                next_context=context, rack_winner=1 if physical.shot == 1 else None,
            )

    search = NineBallMCTS(
        FakeMicro(), simulator=Physics(), rules=Rules(),
        simulations=4, max_depth=depth, root_min_visits=0,
    )
    assert search.search(states[0]) == alternative
    report = search.last_report
    assert report["root_visits"] == [0, 4]
    assert report["root_empirical_q"] == [0.0, -1.0]
    assert report["root_invalid_reasons"] == ["no executable descendant", None]
    assert report["discarded_invalidated_simulations"] == depth - 1
    assert search.last_root.visits == 4
    assert search.last_root.edges[0].child.visits == 0


def test_match_search_bootstraps_next_rack_with_updated_score(monkeypatch):
    goal = NineBallGoal("BREAK", 1)
    board = create_nine_ball_rack(breaker=0, seed=7)
    monkeypatch.setattr(module, "generate_nine_ball_goals", lambda *args, **kw: (goal,))
    observed = []

    def value(inputs):
        observed.append(inputs[0, 72:].tolist())
        return torch.tensor([-0.4])

    class Rules:
        def apply(self, before, physical, context):
            return SimpleNamespace(next_state=board, next_context=context, rack_winner=0)

    search = NineBallMCTS(
        FakeMicro(), simulator=FakePhysics(), rules=Rules(),
        value_network=value, match_aware=True, simulations=1, gamma=0.5,
    )
    position = MatchPosition(2, (0, 0), 0)
    assert search.search(board, match_position=position) == goal
    assert observed == [[1.0, 0.5, 0.5, 1.0, 0.0, 0.0]]
    assert search.last_report["root_empirical_q"] == pytest.approx([0.2])
    search.search(board, match_position=position.after_rack(0))
    assert len(observed) == 1
    assert search.last_report["root_empirical_q"] == [1.0]
