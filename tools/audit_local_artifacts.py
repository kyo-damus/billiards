"""Inventory research artifacts without loading models or deleting files.

Usage:
    python tools/audit_local_artifacts.py --output ~/research/local_artifacts_review.tsv

Classification is conservative; being unreferenced by name NEVER proves
that a model, dataset, or log is safe to delete.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path
import subprocess
import tarfile


REPO = Path(__file__).resolve().parents[1]
EXTENSIONS = {".pt", ".pth", ".npz", ".npy", ".csv"}
CURRENT_MODELS = {
    "goal_conditioned_micro_mixed_curriculum_100k.pt",
    "position_power_residual_mlp_2000.pt",
    "macro_candidate_policy_mixed_v2.pt",
    "macro_candidate_value_mixed_v2.pt",
    "macro_state_value_mixed_v2.pt",
}


def tracked_paths():
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        check=True,
        stdout=subprocess.PIPE,
    )
    return {
        part.decode("utf-8")
        for part in result.stdout.split(b"\0")
        if part
    }


def digest(file_path):
    result = hashlib.sha256()
    with file_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_files(tracked):
    results = {}
    for name in tracked:
        if not name.endswith((".py", ".md", ".rst", ".txt")):
            continue
        path = REPO / name
        if path.is_file() and name != "tools/audit_local_artifacts.py":
            results[name] = path.read_text(encoding="utf-8", errors="replace")
    return results


def category(rel, tracked):
    path = Path(rel)
    if path.name in CURRENT_MODELS:
        return "KEEP_CURRENT_MODEL"
    if rel in tracked:
        return "KEEP_GIT_TRACKED"
    if rel.startswith(("macro_policy_iteration_smoke/", "macro_policy_value_iteration_smoke/")):
        return "REVIEW_ITERATION"
    if rel.startswith(("build/results/", "build/results_muzero/")):
        return "REVIEW_LEGACY_BUILD"
    if path.suffix.lower() in {".npz", ".npy", ".csv"}:
        return "REVIEW_DATASET"
    return "REVIEW_CHECKPOINT"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output TSV; no data deleted")
    parser.add_argument(
        "--backup",
        default=str(Path.home() / "billiards_research_backup_20261109.tar.gz"),
        help="Existing backup archive, checked for member names (not byte identity)",
    )
    args = parser.parse_args()

    tracked = tracked_paths()
    texts = source_files(tracked)
    backup = Path(args.backup).expanduser()
    archived_names = None
    if backup.is_file():
        with tarfile.open(backup, mode="r:gz") as archive:
            archived_names = {member.name.lstrip("./") for member in archive}

    files = sorted(
        path for path in REPO.rglob("*")
        if path.is_file()
        and path.suffix.lower() in EXTENSIONS
        and ".git" not in path.relative_to(REPO).parts
        and "libs" not in path.relative_to(REPO).parts
    )

    rows = []
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        references = [
            name for name, contents in texts.items()
            if rel in contents or path.name in contents
        ]
        # Name mentions are weak evidence; CLI and dynamic paths are not
        # necessarily written literally in source code.
        backup_member = "research/" + rel
        if archived_names is None:
            backup_state = "NO_BACKUP_FOUND"
        elif backup_member in archived_names:
            backup_state = "NAME_IN_BACKUP"
        else:
            backup_state = "NOT_IN_BACKUP"

        rows.append({
            "category": category(rel, tracked),
            "git_tracked": "YES" if rel in tracked else "NO",
            "backup_name_check": backup_state,
            "size_bytes": path.stat().st_size,
            "sha256": digest(path),
            "name_mentions": ";".join(sorted(references)[:12]),
            "relative_path": rel,
        })

    dest = Path(args.output).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "category", "git_tracked", "backup_name_check",
                "size_bytes", "sha256", "name_mentions", "relative_path",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Report: {dest}")
    print(f"Artifacts: {len(rows)}")
    counts = Counter(row["category"] for row in rows)
    for label, count in sorted(counts.items()):
        size = sum(row["size_bytes"] for row in rows if row["category"] == label)
        print(f"  {label}: {count} files, {size / 1048576:.2f} MiB")
    print("Backup member-name check:", dict(Counter(
        row["backup_name_check"] for row in rows
    )))

    hashes = defaultdict(list)
    for row in rows:
        hashes[row["sha256"]].append(row["relative_path"])
    identical = [paths for paths in hashes.values() if len(paths) > 1]
    print(f"Byte-identical groups (SHA-256): {len(identical)}")
    for group in identical[:12]:
        print("  " + " | ".join(group))
    if len(identical) > 12:
        print(f"  ... {len(identical) - 12} more groups; see TSV")
    print("No models were loaded and no existing artifact was deleted.")
    print("Backup check confirms names only, NOT byte-for-byte restoration.")


if __name__ == "__main__":
    main()
