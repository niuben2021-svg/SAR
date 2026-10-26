"""Read-only FAIR-CSAR directory inventory; never infer annotation semantics."""

import argparse
from collections import Counter
from pathlib import Path

from sar_vlm.runtime import timestamp, write_json


def inventory(root: Path, sample_limit: int = 100) -> dict:
    """List file extensions/sizes without opening arrays, archives or annotation contents."""
    root = root.resolve(strict=True)
    counts: Counter = Counter()
    samples, total = [], 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"dataset symlink escapes root: {path}")
        counts[path.suffix.lower() or "<no extension>"] += 1
        total += path.stat().st_size
        if len(samples) < sample_limit:
            samples.append(
                {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}
            )
    return {
        "timestamp": timestamp(),
        "root": str(root),
        "file_count": sum(counts.values()),
        "total_bytes": total,
        "extensions": dict(counts),
        "sample_files": samples,
        "annotation_format": "UNCONFIRMED",
        "complex_encoding": "UNCONFIRMED",
        "license": "UNCONFIRMED",
        "training_data_ready": False,
    }


def main() -> None:
    """Persist a dataset inventory outside the read-only input tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("write inventory outside the raw dataset directory")
    result = inventory(args.root)
    write_json(args.output, result)
    print(f"Inventoried {result['file_count']} files; encoding/annotations remain UNCONFIRMED")


if __name__ == "__main__":
    main()
