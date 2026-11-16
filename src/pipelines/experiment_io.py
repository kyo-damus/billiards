"""Atomic JSON writes and exclusive ownership of an experiment directory."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


@contextmanager
def lock_run(directory):
    with (Path(directory) / ".run.lock").open("a+b") as lock:
        fd = lock.fileno()
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Run is already active: {directory}") from exc
        yield fd


def write_experiment_json(path, data, *, replace=False):
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            temporary.replace(path)
        else:
            # Publish a complete file atomically while preserving create-only semantics.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
