import hashlib
import json

import pytest

from scripts.eval_nine_ball_hierarchical_v3 import MODEL_KEYS, evaluate, parse_args


@pytest.fixture
def evaluation_args(tmp_path):
    base, final = {}, {}
    for key in MODEL_KEYS:
        base_file = tmp_path / f"base_{key}.pt"
        final_file = tmp_path / f"final_{key}.pt"
        base_file.write_bytes(key.encode())
        final_file.write_bytes(key.encode() + b"final")
        base[key] = {
            "path": str(base_file),
            "sha256": hashlib.sha256(base_file.read_bytes()).hexdigest(),
        }
        final[key] = str(final_file)
    manifest = tmp_path / "training.json"
    manifest.write_text(json.dumps({
        "kind": "nine_ball_macro_micro_alternating_v3",
        "status": "completed",
        "source_checkpoints": base,
        "rounds": [{"start_seed": 100, "matches": 2, **final}],
        "settings": {
            "mcts_simulations": 24, "mcts_max_depth": 2, "mcts_c_puct": 1.5,
            "gamma": 0.99, "mcts_root_min_visits": 1, "max_shots_per_rack": 80,
        },
    }))
    return parse_args([
        "--training-manifest", str(manifest), "--pairs", "2", "--start-seed", "200",
        "--output-dir", str(tmp_path / "evaluation"),
    ])


def test_pairs_share_rack_and_exchange_breaker(evaluation_args, monkeypatch):
    import numpy as np
    import scripts.collect_nine_ball_self_play as collector

    initial = []
    # A sweep on seed 200, followed by one completed loss and one interrupted game.
    winners = iter((0, 0, 0, None))

    def fake_play(match, agents, *, max_shots_per_rack):
        assert max_shots_per_rack == 80
        initial.append((match.seed, match.state.current_player, agents, match.state.ball_positions))
        winner = next(winners)
        return {
            "match_finished": winner is not None,
            "match_winner": winner,
            "stop_reason": "race_to_reached" if winner is not None else "invalid_physics",
            "shots": 1,
            "fouls": 0,
        }

    class Physics:
        def restore(self, state):
            pass

    def fake_agent(micro_checkpoint, **kwargs):
        assert kwargs["simulations"] == 24
        assert kwargs["max_depth"] == 2
        return "base" if "base_micro.pt" in micro_checkpoint else "final"

    monkeypatch.setattr(collector, "play_match", fake_play)
    args = evaluation_args
    summary = evaluate(args, agent_factory=fake_agent, simulator_factory=lambda _: Physics())
    assert [(seed, breaker, players) for seed, breaker, players, _ in initial] == [
        (200, 0, ("final", "base")), (200, 1, ("final", "base")),
        (201, 1, ("base", "final")), (201, 0, ("base", "final")),
    ]
    np.testing.assert_array_equal(initial[0][3], initial[1][3])
    np.testing.assert_array_equal(initial[2][3], initial[3][3])
    assert summary["overall"]["final_win_rate_completed"] == pytest.approx(2 / 3)
    assert summary["overall"]["unfinished"] == 1
    assert summary["paired_results"] == {
        "final_won_both": 1, "base_won_both": 0, "split": 0, "unfinished": 1,
    }
    assert len(list(args.output_dir.glob("seed_*.json"))) == 4
    original = (args.output_dir / "summary.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        evaluate(args)
    assert (args.output_dir / "summary.json").read_bytes() == original


def test_rejects_training_seeds_and_changed_starting_models(evaluation_args):
    args = evaluation_args
    args.start_seed = 102
    with pytest.raises(ValueError, match="overlap"):
        evaluate(args)
    args.start_seed = 200
    (args.training_manifest.parent / "base_micro.pt").write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="differs from the training manifest"):
        evaluate(args)
    assert not args.output_dir.exists()
