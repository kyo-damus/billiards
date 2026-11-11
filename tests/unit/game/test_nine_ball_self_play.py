"""Fast, GPU-free action-provider and trajectory-runner tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.game.match import MatchConfig, RaceToMatch
from src.game.nine_ball_agents import GeometryNineBallAgent
from src.game.nine_ball_session import create_nine_ball_rack
from scripts.collect_nine_ball_self_play import (
    collect_matches, play_match, rack_state_json,
)


def test_geometry_agent_returns_physical_five_parameter_shot():
    agent = GeometryNineBallAgent()
    rack = create_nine_ball_rack(breaker=0, seed=2)
    plan = agent.choose_shot(rack)
    assert plan.target_ball == 1
    assert plan.goal_label == "BREAK_HEURISTIC"
    assert plan.shot.speed_m_s == 2.5
    assert len(plan.shot.to_fastfiz_kwargs()) == 5


def test_geometry_agent_places_ball_after_foul():
    from dataclasses import replace
    rack = create_nine_ball_rack(breaker=0, seed=5)
    rack = replace(rack, ball_in_hand=True)
    positions = GeometryNineBallAgent().cue_placement_candidates(rack)
    assert positions
    assert 0 < positions[0][0] < 1.116


def test_geometry_agent_does_not_select_shot_with_ball_in_hand():
    from dataclasses import replace
    rack = replace(create_nine_ball_rack(breaker=0, seed=2), ball_in_hand=True)
    with pytest.raises(ValueError, match="placed"):
        GeometryNineBallAgent().choose_shot(rack)


class OneRackFakeMatch:
    """One action ends a rack and match deterministically."""
    def __init__(self, seed=5, winner=1):
        self.state = create_nine_ball_rack(breaker=0, seed=seed)
        self.match = RaceToMatch(MatchConfig(race_to=1, rule_id="nine_ball"))
        self.rack = SimpleNamespace(last_physical=None)
        self.rack_history = []
        self.winner = winner

    def step(self, *, target_ball, shot):
        self.match.finish_rack(self.winner)
        return SimpleNamespace(
            next_state=self.state,
            foul=False,
            foul_reasons=(),
            newly_pocketed=(9,),
            rack_winner=self.winner,
            turn_changed=False,
        )


class EndlessFakeMatch(OneRackFakeMatch):
    def step(self, *, target_ball, shot):
        return SimpleNamespace(
            next_state=self.state,
            foul=False,
            foul_reasons=(),
            newly_pocketed=(),
            rack_winner=None,
            turn_changed=True,
        )


def test_one_completed_match_is_logged_with_actual_winner():
    result = play_match(
        OneRackFakeMatch(), (GeometryNineBallAgent(), GeometryNineBallAgent()),
        max_shots_per_rack=5,
    )
    assert result["match_finished"]
    assert result["match_winner"] == 1
    assert result["shots"] == 1
    assert result["trajectory"][0]["chosen_action"]["target_ball"] == 1
    assert result["trajectory"][0]["chosen_action"]["goal_v3"] is None
    assert result["trajectory"][0]["state"]["ball_numbers"] == list(range(10))
    assert "rule_context" in result["trajectory"][0]


def test_turn_limit_keeps_unfinished_match_without_fabricated_winner():
    result = play_match(
        EndlessFakeMatch(), (GeometryNineBallAgent(), GeometryNineBallAgent()),
        max_shots_per_rack=3,
    )
    assert not result["match_finished"]
    assert result["match_winner"] is None
    assert result["stop_reason"] == "rack_shot_limit"
    assert len(result["trajectory"]) == 3


def test_output_files_separate_and_no_overwrite(tmp_path):
    import json

    out = tmp_path / "result"
    result = collect_matches(
        matches=2, race_to=1, start_seed=300,
        max_shots_per_rack=3, output_dir=out,
        match_factory=lambda seed: OneRackFakeMatch(seed),
    )
    assert result["finished_matches"] == 2
    assert result["unfinished_matches"] == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "match_seed_300.json", "match_seed_302.json", "summary.json"
    ]
    assert json.loads((out / "match_seed_300.json").read_text())["match_winner"] == 1
    with pytest.raises(FileExistsError, match="Refusing"):
        collect_matches(
            matches=1, race_to=1, start_seed=400,
            max_shots_per_rack=3, output_dir=out,
            match_factory=lambda seed: OneRackFakeMatch(seed),
        )


def test_invalid_match_limits_refused_before_output(tmp_path):
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        collect_matches(
            matches=1, race_to=0, start_seed=1,
            max_shots_per_rack=10, output_dir=out,
            match_factory=OneRackFakeMatch,
        )
    assert not out.exists()


class EvidencePreservingFakeMatch(OneRackFakeMatch):
    """Simulate a new rack replacing the old one immediately after a win."""

    def step(self, *, target_ball, shot):
        evidence = SimpleNamespace(
            first_contact_ball=1,
            precondition_code=0,
            events=(
                SimpleNamespace(
                    time_s=0.1, kind="BALL_COLLISION", ball1=0, ball2=1
                ),
            ),
        )
        self.last_physical = evidence
        self.rack = SimpleNamespace(last_physical=None)
        return super().step(target_ball=target_ball, shot=shot)


def test_terminal_rack_physical_evidence_survives_rack_replacement():
    result = play_match(
        EvidencePreservingFakeMatch(),
        (GeometryNineBallAgent(), GeometryNineBallAgent()),
        max_shots_per_rack=1,
    )
    row = result["trajectory"][0]
    assert row["physical"]["first_contact_ball"] == 1
    assert row["physical"]["events"][0]["kind"] == "BALL_COLLISION"



def test_learned_agent_trajectories_are_not_mislabeled_as_geometry(tmp_path):
    class FakeLearnedAgent(GeometryNineBallAgent):
        pass
    folder = tmp_path / "learned"
    outcome = collect_matches(
        matches=1, race_to=1, start_seed=400,
        max_shots_per_rack=2, output_dir=folder,
        agent_factory=FakeLearnedAgent,
        match_factory=lambda seed: OneRackFakeMatch(seed),
    )
    assert outcome["run_type"] == "learned_micro_self_play"
    assert outcome["per_match"][0]["players"] == [
        "FakeLearnedAgent", "FakeLearnedAgent",
    ]
    import json
    payload = json.loads((folder / "match_seed_400.json").read_text())
    assert payload["run_type"] == "learned_micro_self_play"


def test_exact_versioned_chosen_macro_goal_is_logged():
    from dataclasses import replace
    from src.game.nine_ball_encoding_v3 import (
        NineBallGoal, encode_goal_v3,
    )

    class GoalProducingAgent(GeometryNineBallAgent):
        def choose_shot(self, state, context=None):
            plan = super().choose_shot(state, context)
            return replace(
                plan, macro_goal=NineBallGoal("BREAK", target_ball=plan.target_ball),
            )

    outcome = play_match(
        OneRackFakeMatch(),
        (GoalProducingAgent(), GoalProducingAgent()),
        max_shots_per_rack=1,
    )
    chosen = outcome["trajectory"][0]["chosen_action"]
    assert len(chosen["goal_v3"]) == 36
    np.testing.assert_allclose(
        chosen["goal_v3"], encode_goal_v3(NineBallGoal("BREAK", 1)),
    )


def test_search_root_visits_saved_with_executed_goal():
    from dataclasses import replace
    from src.game.nine_ball_encoding_v3 import NineBallGoal, encode_goal_v3

    goal = NineBallGoal("BREAK", target_ball=1)
    diagnostics = {
        "simulations": 2,
        "max_depth": 1,
        "root_candidates": 1,
        "chosen_index": 0,
        "root_visits": [2],
        "root_priors": [1.0],
        "root_empirical_q": [0.0],
        "goal_v3": [encode_goal_v3(goal).tolist()],
        "uses_learned_state_v": False,
        "uses_selected_action_q": False,
    }

    class FakeMCTSAgent(GeometryNineBallAgent):
        def choose_shot(self, state, context=None):
            return replace(
                super().choose_shot(state, context),
                macro_goal=goal,
                macro_search=diagnostics,
                goal_label="BREAK_MCTS_V3",
            )

    result = play_match(
        OneRackFakeMatch(),
        (FakeMCTSAgent(), FakeMCTSAgent()),
        max_shots_per_rack=1,
    )
    row = result["trajectory"][0]["chosen_action"]
    assert row["macro_search"] == diagnostics
    assert row["goal_v3"] == encode_goal_v3(goal).tolist()
    assert row["macro_search"]["chosen_index"] == 0


def test_resume_skips_saved_seed_and_collects_only_remaining_matches(tmp_path):
    import json
    out = tmp_path / "unfinished"
    collect_matches(
        matches=1, race_to=1, start_seed=700,
        max_shots_per_rack=3, output_dir=out,
        match_factory=lambda seed: OneRackFakeMatch(seed),
    )
    original = (out / "match_seed_700.json").read_bytes()
    # Simulate interruption after the first saved match, before final summary.
    (out / "summary.json").unlink()
    outcome = collect_matches(
        matches=2, race_to=1, start_seed=700,
        max_shots_per_rack=3, output_dir=out,
        match_factory=lambda seed: OneRackFakeMatch(seed),
        resume=True,
    )
    assert outcome["matches"] == 2
    assert (out / "match_seed_700.json").read_bytes() == original
    assert (out / "match_seed_702.json").is_file()
    assert (out / "summary.json").is_file()
    assert json.loads((out / "summary.json").read_text())["finished_matches"] == 2

    with pytest.raises(FileExistsError, match="completed collector summary"):
        collect_matches(
            matches=2, race_to=1, start_seed=700,
            max_shots_per_rack=3, output_dir=out,
            match_factory=lambda seed: OneRackFakeMatch(seed), resume=True,
        )


def test_resume_refuses_unexpected_saved_match_seed(tmp_path):
    out = tmp_path / "bad"
    out.mkdir()
    (out / "match_seed_999.json").write_text("{}")
    with pytest.raises(ValueError, match="Unexpected seed"):
        collect_matches(
            matches=2, race_to=1, start_seed=700,
            max_shots_per_rack=3, output_dir=out,
            match_factory=lambda seed: OneRackFakeMatch(seed), resume=True,
        )


def test_nonphysical_match_is_unfinished_not_an_invented_foul():
    from src.game.nine_ball_physics_guard import InvalidPhysicalTransition

    class BrokenPhysicsAgent(GeometryNineBallAgent):
        def choose_shot(self, state, context=None):
            raise InvalidPhysicalTransition("Bad Shot Exception: root unavailable")

    outcome = play_match(
        OneRackFakeMatch(), (BrokenPhysicsAgent(), BrokenPhysicsAgent()),
        max_shots_per_rack=3,
    )
    assert outcome["match_finished"] is False
    assert outcome["match_winner"] is None
    assert outcome["stop_reason"] == "invalid_physics"
    assert outcome["shots"] == 0
    assert outcome["fouls"] == 0
    assert "root unavailable" in outcome["physics_failure"]["error"]
