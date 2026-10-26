#!/usr/bin/env python3
"""Archive reviewed source paths only; exclude Git credentials, data, weights and outputs."""

import argparse
import hashlib
import tarfile
from pathlib import Path


def main() -> None:
    """Package the current worktree implementation, including uncommitted source files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        parser.error("refusing to overwrite source package")
    include = [
        "src",
        "tests",
        "configs",
        "scripts",
        "deploy",
        "docs",
        "pyproject.toml",
        "README.md",
        "AGENTS.md",
        "PROJECT_CONTEXT.md",
        "TASKS.md",
        ".gitignore",
        ".dockerignore",
        "requirements-cpu-lock.txt",
        "reports/runs/offline_delivery_report.md",
        "reports/runs/qwen3vl4b_delivery_report.md",
    ]
    paths = []
    for name in include:
        base = root / name
        if not base.exists() and name.startswith("reports/"):
            continue
        paths.extend(p for p in base.rglob("*") if p.is_file()) if base.is_dir() else paths.append(
            base
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for path in sorted(paths):
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f"unexpected source symlink: {relative}")
            archive.add(path, arcname="sar-vlm/" + relative.as_posix(), recursive=False)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    Path(str(args.output) + ".sha256").write_text(f"{digest}  {args.output.name}\n")
    print(f"Packaged {args.output} (source only, no weights/data/image)")


if __name__ == "__main__":
    main()
