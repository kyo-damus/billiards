"""Prepare a reproducible experiment directory without running an experiment.

Configuration JSON is a human-readable snapshot only. Existing training and
evaluation scripts DO NOT automatically consume this file.

Example:
    python tools/prepare_run.py \
        --config configs/smoke/mixed_one_turn.example.json \
        --id 20261109_mixed_smoke_01
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def git_output(repo_root, *args):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode:
        return None
    return result.stdout.strip()


def validate_config(contents):
    try:
        config = json.loads(contents)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Config must be UTF-8 JSON.") from exc
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object.")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Config schema_version must be 1.")
    if not isinstance(config.get("experiment_type"), str) or not config["experiment_type"].strip():
        raise ValueError("Config requires nonempty experiment_type.")
    if not isinstance(config.get("entrypoint"), str) or not config["entrypoint"].strip():
        raise ValueError("Config requires nonempty entrypoint.")
    return config


def prepare_run(config_path, run_id, run_root, repo_root=PROJECT_ROOT):
    """Create a fresh run folder; never overwrite or execute a model."""
    if not RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("Run ID must be a safe single directory name (1-80 characters).")

    config_path = Path(config_path)
    config_bytes = config_path.read_bytes()
    validate_config(config_bytes.decode("utf-8"))
    source_hash = hashlib.sha256(config_bytes).hexdigest()

    run_root = Path(run_root)
    run_dir = run_root / run_id
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(f"Run already exists; refusing overwrite: {run_dir}")

    commit = git_output(repo_root, "rev-parse", "HEAD")
    branch = git_output(repo_root, "branch", "--show-current")
    status = git_output(repo_root, "status", "--porcelain")
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config_path),
        "config_sha256": source_hash,
        "git_commit": commit,
        "git_branch": branch,
        "git_worktree_dirty": None if status is None else bool(status),
        "python_version": platform.python_version(),
        "note": (
            "Directory preparation only. Training/evaluation was not run. "
            "Record the actual command and outputs separately."
        ),
    }

    run_dir.mkdir(parents=True, exist_ok=False)
    # In case of a filesystem error, leave partial data visible rather than
    # silently deleting anything; rerun with a new ID after inspection.
    (run_dir / "config.json").write_bytes(config_bytes)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "outputs").mkdir()
    (run_dir / "logs").mkdir()
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--id", required=True, help="Unique experiment run ID")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runs",
        help="Root directory for generated run folders",
    )
    args = parser.parse_args()
    try:
        directory = prepare_run(args.config, args.id, args.root)
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(f"Prepared run directory: {directory}")
    print("Recorded config and git metadata; no training was executed.")


if __name__ == "__main__":
    main()
