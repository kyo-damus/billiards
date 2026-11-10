"""Integration-runner contracts using mocked training/evaluation (no FastFiz)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_full_hierarchical_experiment import (
    final_checkpoints,
    parse_args,
    run_experiment,
)


def args_for(tmp_path, *more):
    root = tmp_path / "model"
    root.mkdir(exist_ok=True)
    checkpoints = [root / name for name in ("base.pt", "residual.pt", "policy.pt", "q.pt", "v.pt")]
    for path in checkpoints:
        path.write_text("stub", encoding="utf-8")
    return parse_args([
        "--base-checkpoint", str(checkpoints[0]),
        "--residual-checkpoint", str(checkpoints[1]),
        "--candidate-policy-checkpoint", str(checkpoints[2]),
        "--candidate-value-checkpoint", str(checkpoints[3]),
        "--state-value-checkpoint", str(checkpoints[4]),
        "--output-dir", str(tmp_path / "experiment"),
        "--iterations", "2",
        "--train-games-per-iteration", "2",
        "--eval-games", "2",
        "--start-seed", "100",
        "--eval-start-seed", "200",
        *more,
    ])


def test_refuses_overlapping_train_eval_seeds_before_creating_directory(tmp_path):
    args = args_for(tmp_path, "--eval-start-seed", "101")
    with pytest.raises(ValueError, match="seed ranges overlap"):
        run_experiment(args, state_factory=lambda seed: seed)
    assert not args.output_dir.exists()


def test_refuses_existing_experiment_directory_before_training(tmp_path):
    args = args_for(tmp_path)
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        run_experiment(args, state_factory=lambda seed: seed)


def test_final_checkpoints_keep_latest_critics_when_last_update_skipped(tmp_path):
    old_q = tmp_path / "q0.pt"
    old_v = tmp_path / "v0.pt"
    policy = tmp_path / "p2.pt"
    q1 = tmp_path / "q1.pt"
    v1 = tmp_path / "v1.pt"
    for path in (policy, q1, v1):
        path.write_text("stub", encoding="utf-8")
    records = [
        SimpleNamespace(output_policy=policy, output_q=q1, output_v=v1),
        SimpleNamespace(output_policy=policy, output_q=None, output_v=None),
    ]
    assert final_checkpoints(records, old_q, old_v) == (
        str(policy), str(q1), str(v1)
    )


def test_run_experiment_trains_then_compares_heldout_seeds_with_full_v(tmp_path):
    args = args_for(tmp_path)
    calls = []
    root = Path(args.output_dir)
    mock_policy = root / "training" / "iteration_001" / "candidate_policy.pt"
    mock_q = root / "training" / "iteration_001" / "candidate_q.pt"
    mock_v = root / "training" / "iteration_001" / "state_value.pt"

    def train(a):
        calls.append(("train", a.games_per_iteration, a.simulations, a.output_dir))
        for path in (mock_policy, mock_q, mock_v):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("stub", encoding="utf-8")
        return [
            SimpleNamespace(
                index=0, visited_roots=3, candidate_rows=21, complete_games=2,
                output_policy=mock_policy, output_q=mock_q, output_v=mock_v,
            ),
            SimpleNamespace(
                index=1, visited_roots=4, candidate_rows=28, complete_games=0,
                output_policy=mock_policy, output_q=None, output_v=None,
            ),
        ]

    def agent_factory(a, device):
        calls.append(("agent", a.candidate_policy_checkpoint,
                      a.candidate_value_checkpoint, a.state_value_checkpoint,
                      a.simulations))
        return object()

    def collect(agent, factory, **kwargs):
        calls.append(("collect", kwargs["start_seed"], kwargs["games"],
                      kwargs["output_dir"], factory(200)))
        return SimpleNamespace(saved_games=2)

    def aggregate(paths):
        calls.append(("report", len(paths), paths[0].name))
        return {"games": 2, "completed_games": 1, "score_delta": [1.0, 0.0]}

    result = run_experiment(
        args, iterate=train, collect=collect, agent_factory=agent_factory,
        state_factory=lambda seed: seed, report=aggregate,
    )
    assert calls[0][:3] == ("train", 2, 50)
    agent_calls = [x for x in calls if x[0] == "agent"]
    assert len(agent_calls) == 2
    assert agent_calls[0][3] == args.state_value_checkpoint
    assert agent_calls[1][3] == str(mock_v)
    assert agent_calls[1][2] == str(mock_q)
    assert len([x for x in calls if x[0] == "collect"]) == 2
    assert all(x[1:3] == (200, 2) for x in calls if x[0] == "collect")
    assert result["training"][-1]["state_v"] is None
    assert result["evaluated_checkpoints"]["after_training"]["state_v"] == str(mock_v)
    assert (root / "experiment_summary.json").is_file()
