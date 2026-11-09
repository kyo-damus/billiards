"""Safely quarantine 107 obsolete build checkpoints after SHA-256 backup verification.

Default action is READ-ONLY verification:
    python tools/quarantine_legacy_build_artifacts.py

Move only after inspection:
    python tools/quarantine_legacy_build_artifacts.py --apply

To move the exact same files back:
    python tools/quarantine_legacy_build_artifacts.py --restore

Never deletes a model, changes a checkpoint, or overwrites a path.
Requires the local_artifacts_review.tsv and the November 9 tar backup.
"""

import argparse
import csv
import hashlib
from pathlib import Path
import re
import tarfile


ROOT = Path(__file__).resolve().parents[1]
BACKUP = Path.home() / "billiards_research_backup_20261109.tar.gz"
DEST = ROOT.parent / "billiards_refactor_quarantine_20261109"
MANIFEST = "_legacy_build_quarantine_manifest.tsv"

EXPECTED = {
    f"build/results/sac_actor_ep{i}.pth"
    for i in range(1000, 100001, 1000)
} | {
    "build/results/sac_actor_final.pth",
    "build/results/sac_critic_final.pth",
    "build/results_muzero/dyn_net_latest.pth",
    "build/results_muzero/pred_net_latest.pth",
    "build/results_muzero/rep_net_latest.pth",
    "build/mcts_experience.pt",
    "build/policy_model.pth",
}

assert len(EXPECTED) == 107


def sha256_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def sha256_file(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def check_location(quarantine):
    if quarantine.resolve() == ROOT.resolve():
        raise ValueError("Quarantine destination is the repository.")
    if quarantine.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Quarantine must be outside the repository.")


def load_tsv(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    result = {}
    for row in rows:
        rel = row.get("relative_path", "")
        if rel in result:
            raise ValueError(f"Duplicate artifact path: {rel}")
        result[rel] = row
    return result


def selected_rows(report):
    all_rows = load_tsv(report)
    missing = EXPECTED - all_rows.keys()
    if missing:
        raise ValueError(f"Audit TSV is missing: {sorted(missing)}")
    selected = {path: all_rows[path] for path in sorted(EXPECTED)}
    for path, row in selected.items():
        if row["git_tracked"] != "NO":
            raise ValueError(f"Refusing to move tracked file: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError(f"Invalid SHA256 in audit TSV: {path}")
        if row["backup_name_check"] != "NAME_IN_BACKUP":
            raise ValueError(f"No previous backup name match: {path}")
        expected_category = (
            "REVIEW_LEGACY_BUILD"
            if path.startswith(("build/results/", "build/results_muzero/"))
            else "REVIEW_CHECKPOINT"
        )
        if row["category"] != expected_category:
            raise ValueError(f"Unexpected audit category: {path}")
    return selected


def verify_files(rows, folder):
    for rel, row in rows.items():
        path = folder / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symlink artifact: {path}")
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Size changed since audit: {path}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"SHA256 changed since audit: {path}")


def verify_backup(rows, archive_path):
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError(f"Backup missing or symlink: {archive_path}")
    members_needed = {"research/" + rel: rel for rel in rows}
    found = set()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            rel = members_needed.get(member.name.lstrip("./"))
            if rel is None:
                continue
            if rel in found or not member.isfile():
                raise ValueError(f"Invalid/duplicate tar member: {rel}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read archived file: {rel}")
            with stream:
                archived_hash = sha256_stream(stream)
            if archived_hash != rows[rel]["sha256"]:
                raise ValueError(f"BACKUP CONTENT MISMATCH: {rel}")
            found.add(rel)
    if found != set(rows):
        raise ValueError(f"Missing tar members: {sorted(set(rows) - found)}")


def preflight_destinations(rows, source, dest):
    for rel in rows:
        target = dest / rel
        if target.exists() or target.is_symlink():
            raise ValueError(f"Refusing to overwrite existing file: {target}")
        if not (source / rel).is_file():
            raise ValueError(f"Source missing: {source / rel}")


def move_atomically_per_file(rows, source, dest):
    """Best-effort rollback if an individual rename or manifest write fails."""
    moved = []
    try:
        for rel in rows:
            old, new = source / rel, dest / rel
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
            moved.append((old, new))
    except BaseException:
        for old, new in reversed(moved):
            if new.exists() and not old.exists():
                old.parent.mkdir(parents=True, exist_ok=True)
                new.rename(old)
        raise
    return moved


def save_manifest(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, delimiter="\t",
            fieldnames=["relative_path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        for rel, row in rows.items():
            writer.writerow({
                "relative_path": rel,
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Move verified files")
    mode.add_argument("--restore", action="store_true", help="Move files back")
    p.add_argument("--report", default=str(ROOT / "local_artifacts_review.tsv"))
    p.add_argument("--backup", default=str(BACKUP))
    p.add_argument("--quarantine", default=str(DEST))
    args = p.parse_args()

    quarantine = Path(args.quarantine).expanduser().resolve()
    check_location(quarantine)
    manifest = quarantine / MANIFEST

    if args.restore:
        if not manifest.is_file():
            raise ValueError(f"Missing quarantine manifest: {manifest}")
        rows = load_tsv(manifest)
        if set(rows) != EXPECTED:
            raise ValueError("Quarantine manifest paths do not match whitelist.")
        verify_files(rows, quarantine)
        preflight_destinations(rows, quarantine, ROOT)
        move_atomically_per_file(rows, quarantine, ROOT)
        manifest.rename(quarantine / (MANIFEST + ".restored"))
        print(f"Restored {len(rows)} files into {ROOT}")
        return

    rows = selected_rows(Path(args.report).expanduser())
    verify_files(rows, ROOT)
    verify_backup(rows, Path(args.backup).expanduser())
    size = sum(int(row["size_bytes"]) for row in rows.values())
    print(f"All {len(rows)} legacy files match local audit AND tar backup bytes.")
    print(f"Total: {size / 1048576:.2f} MiB; quarantine: {quarantine}")
    if not args.apply:
        print("READ-ONLY verification completed. No files moved.")
        print("Run with --apply to move the 107 verified files.")
        return

    if manifest.exists():
        raise ValueError(f"Quarantine already contains manifest: {manifest}")
    preflight_destinations(rows, ROOT, quarantine)
    moved = move_atomically_per_file(rows, ROOT, quarantine)
    try:
        save_manifest(manifest, rows)
    except BaseException:
        for old, new in reversed(moved):
            old.parent.mkdir(parents=True, exist_ok=True)
            new.rename(old)
        raise
    print(f"Quarantined {len(rows)} files. Nothing deleted.")
    print(f"Restore anytime using: python tools/quarantine_legacy_build_artifacts.py --restore")


if __name__ == "__main__":
    main()
