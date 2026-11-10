"""Full native game path: bounded autonomous 9-ball with geometry placeholder."""

from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.match import MatchConfig
from src.game.nine_ball_agents import GeometryNineBallAgent
from src.game.nine_ball_session import NineBallMatchSession
from scripts.collect_nine_ball_self_play import play_match


def test_native_three_shot_autonomous_match_recording():
    match = NineBallMatchSession(
        simulator_factory=MultiBallFastFizSimulator,
        match_config=MatchConfig(race_to=1, rule_id="nine_ball"),
        seed=420,
    )
    result = play_match(
        match, (GeometryNineBallAgent(), GeometryNineBallAgent()),
        max_shots_per_rack=3,
    )
    assert 1 <= result["shots"] <= 3
    assert len(result["trajectory"]) == result["shots"]
    assert result["match_winner"] in (None, 0, 1)
    for shot in result["trajectory"]:
        assert shot["chosen_action"]["target_ball"] in range(1, 10)
        assert len(shot["state"]["positions"]) == 10
        assert "events" in shot["physical"]
        assert shot["physical"]["precondition_code"] is not None
