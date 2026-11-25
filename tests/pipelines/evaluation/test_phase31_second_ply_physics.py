"""CPU-only Phase31 second-ply physical audit report tests."""
from scripts.experiments.diagnose_phase31_second_ply_physics import summarize


def _reply(*, foul, potted, next_player):
    return {
        "visits": 1, "second_shooter": 1,
        "physical_result": {
            "available": True,
            "foul": foul,
            "target_ball_pocketed": potted,
            "next_player": next_player,
        },
    }


def test_summarize_counts_only_physical_replays_not_unavailable():
    row = {
        "models": {
            "base": {
                "C_choice": {
                    "first_foul": False, "second_replies": [
                        _reply(foul=False, potted=True, next_player=1),
                    ],
                },
                "Cprime_choice": {
                    "first_foul": True, "second_replies": [
                        _reply(foul=True, potted=False, next_player=0),
                        {
                            "visits": 1,
                            "second_shooter": 1,
                            "physical_result": {
                                "available": False,
                                "reason": "invalid_native_transition",
                            },
                        },
                    ],
                },
            },
            "final": {
                "C_choice": {
                    "first_foul": False, "second_replies": [],
                },
                "Cprime_choice": {
                    "first_foul": True, "second_replies": [
                        _reply(foul=False, potted=False, next_player=0),
                    ],
                },
            },
        }
    }
    report = summarize([row])
    assert report["base:C_choice"]["second_shot_target_pots"] == 1
    assert report["base:C_choice"]["second_shot_keeps_turn"] == 1
    assert report["base:Cprime_choice"]["first_shot_fouls"] == 1
    assert report["base:Cprime_choice"]["second_reply_candidates_simulated"] == 1
    assert report["base:Cprime_choice"]["second_shot_fouls"] == 1
    assert report["final:Cprime_choice"]["second_shot_keeps_turn"] == 0
    assert report["final:C_choice"]["second_reply_candidates_simulated"] == 0
