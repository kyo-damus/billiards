"""Tests for Phase-1 experiment run scaffolding; no physics or training."""

import hashlib
import json

import pytest

from tools.prepare_run import prepare_run


def write_config(path):
    data = {
        "schema_version": 1,
        "experiment_type": "mixed_hierarchical_smoke",
        "entrypoint": "scripts.smoke_mixed_hierarchical_rollout",
        "seed": 94000,
        "model_contract": {
            "candidate_dim": 31,
            "candidate_encoding_version": 2,
            "turn_aware": True,
        },
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def test_prepare_run_records_config_and_creates_output_dirs(tmp_path):
    config = tmp_path / "smoke.json"
    source = write_config(config)
    root = tmp_path / "artifacts" / "runs"
    target = prepare_run(config, "20261109_smoke_01", root, repo_root=tmp_path)

    assert target == root / "20261109_smoke_01"
    assert json.loads((target / "config.json").read_text()) == source
    metadata = json.loads((target / "metadata.json").read_text())
    assert metadata["run_id"] == "20261109_smoke_01"
    assert metadata["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert metadata["git_commit"] is None  # tmp_path is not a git repository
    assert metadata["git_worktree_dirty"] is None
    assert (target / "outputs").is_dir()
    assert (target / "logs").is_dir()
    assert sorted(p.name for p in target.iterdir()) == [
        "config.json", "logs", "metadata.json", "outputs",
    ]


def test_prepare_run_never_overwrites_existing_run(tmp_path):
    config = tmp_path / "smoke.json"
    write_config(config)
    root = tmp_path / "runs"
    target = prepare_run(config, "smoke_01", root, repo_root=tmp_path)
    marker = target / "outputs" / "keep.pt"
    marker.write_bytes(b"not-a-real-checkpoint")

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        prepare_run(config, "smoke_01", root, repo_root=tmp_path)

    assert marker.read_bytes() == b"not-a-real-checkpoint"


@pytest.mark.parametrize("invalid_id", ["..", ".", "../escape", "bad/name", "bad name", ""])
def test_prepare_run_rejects_unsafe_ids(tmp_path, invalid_id):
    config = tmp_path / "smoke.json"
    write_config(config)
    root = tmp_path / "runs"

    with pytest.raises(ValueError, match="safe single directory"):
        prepare_run(config, invalid_id, root, repo_root=tmp_path)
    assert not root.exists()


def test_prepare_run_rejects_invalid_config_without_side_effects(tmp_path):
    config = tmp_path / "invalid.json"
    config.write_text('{"schema_version": 2, "experiment_type": "x", "entrypoint": "y"}')
    root = tmp_path / "runs"

    with pytest.raises(ValueError, match="schema_version"):
        prepare_run(config, "exp_01", root, repo_root=tmp_path)
    assert not root.exists()
