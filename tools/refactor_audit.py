"""Read-only local audit of Python modules and tests for refactoring.

Run from repository root:
    python tools/refactor_audit.py | tee refactor_audit.txt

The report is heuristic: dynamic imports, CLI usage, saved checkpoints,
historical experiments, and runtime reflection can create false positives.
Nothing is deleted or modified by this program.
"""

import ast
from collections import Counter, defaultdict
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src/", "scripts/", "tests/", "build/")
MODEL_SUFFIXES = {".pt", ".pth", ".npz", ".npy", ".csv"}


def tracked_files():
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return {p.decode("utf-8") for p in result.stdout.split(b"\0") if p}


def python_module(path):
    return path.removesuffix(".py").replace("/", ".")


def resolve_import(module_name, modules):
    if module_name in modules:
        return modules[module_name]
    return None


def scan_python(tracked):
    paths = sorted(
        p for p in tracked
        if p.endswith(".py")
        and p.startswith(SOURCE_DIRS)
        and (ROOT / p).is_file()
    )
    modules = {python_module(p): p for p in paths}
    imported_by = defaultdict(set)
    test_counts = {}
    syntax_errors = []

    for path in paths:
        try:
            tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as exc:
            syntax_errors.append(f"{path}: {exc}")
            continue

        if path.startswith("tests/"):
            test_counts[path] = sum(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
                for node in ast.walk(tree)
            )

        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.module:
                    pieces = python_module(path).split(".")[:-node.level]
                    base = ".".join(pieces + node.module.split("."))
                elif node.level:
                    base = ".".join(
                        python_module(path).split(".")[:-node.level]
                    )
                else:
                    base = node.module or ""
                names = [base]
                names.extend(
                    base + "." + item.name
                    for item in node.names
                    if base and item.name != "*"
                )
            for name in names:
                match = resolve_import(name, modules)
                if match is not None and match != path:
                    imported_by[match].add(path)

                # Some historical build scripts import sibling modules
                # without the 'build.' prefix.
                if "." not in name and path.startswith("build/"):
                    sibling = "build/" + name + ".py"
                    if sibling in paths and sibling != path:
                        imported_by[sibling].add(path)

    return paths, imported_by, test_counts, syntax_errors


def referenced_in_docs(path, doc_text):
    return any(
        text in doc_text
        for text in (path, path.removesuffix(".py"))
    )


def print_section(label, rows, limit=100):
    print(f"\n=== {label} ({len(rows)}) ===")
    for row in rows[:limit]:
        print(row)
    if len(rows) > limit:
        print(f"... {len(rows) - limit} additional entries omitted")


def main():
    tracked = tracked_files()
    paths, imported_by, test_counts, errors = scan_python(tracked)
    docs = []
    for path in sorted(tracked):
        if path.endswith((".md", ".rst", ".txt")):
            file = ROOT / path
            if file.is_file():
                docs.append(file.read_text(encoding="utf-8", errors="replace"))
    doc_text = "\n".join(docs)

    print("=== Read-only refactoring audit ===")
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch:", branch)
    print("HEAD:", head)
    print("tracked python files:", len(paths))
    print("tracked tests:", len(test_counts))
    print("test functions:", sum(test_counts.values()))

    src_unreferenced = sorted(
        path for path in paths
        if path.startswith("src/")
        and not imported_by[path]
        and not referenced_in_docs(path, doc_text)
    )
    scripts_unreferenced = sorted(
        path for path in paths
        if path.startswith("scripts/")
        and not imported_by[path]
        and not referenced_in_docs(path, doc_text)
    )
    build_candidates = sorted(
        path for path in paths
        if path.startswith("build/")
    )
    print_section(
        "SRC modules without static imports / direct docs references"
        " (REVIEW, not confirmed dead)",
        src_unreferenced,
    )
    print_section(
        "Standalone/unreferenced scripts (CLI entrypoints may be used)",
        scripts_unreferenced,
    )
    print_section(
        "Legacy build Python files with known static importers",
        [
            f"{path} | importers: "
            + (", ".join(sorted(imported_by[path])) or "none")
            for path in build_candidates
        ],
    )

    smallest = sorted(test_counts.items(), key=lambda item: (item[1], item[0]))
    print_section(
        "Test files with zero to two test functions (not deletion advice)",
        [f"{n}: {path}" for path, n in smallest if n <= 2],
    )
    print_section("Python syntax/read errors", errors)

    artifacts = [
        p for p in ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES
        and ".git" not in p.parts
        and "libs" not in p.relative_to(ROOT).parts
    ]
    group = Counter(
        ("git-tracked" if p.relative_to(ROOT).as_posix() in tracked
         else "not-tracked", p.suffix.lower())
        for p in artifacts
    )
    print("\n=== Artifact summary (no files modified) ===")
    for (kind, ext), n in sorted(group.items()):
        size = sum(
            p.stat().st_size for p in artifacts
            if p.suffix.lower() == ext
            and (("git-tracked" if p.relative_to(ROOT).as_posix() in tracked
                  else "not-tracked") == kind)
        )
        print(f"{kind:12} {ext:5} {n:4} files / {size / 1048576:.2f} MiB")

    print("\nNOTE: Static findings do NOT establish whether a file is safe to delete.")
    print("No file, data, model, or Git reference was modified.")


if __name__ == "__main__":
    main()
