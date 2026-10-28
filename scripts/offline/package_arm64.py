"""Verify the saved archive stream and produce its full SHA256 without joining parts."""

import hashlib
from pathlib import Path

root = Path("release-assets")
parts = sorted(root.glob("sar-vlm-qwen3vl4b-arm64-cu126.tar.gz.part*"))
assert parts
assert [p.suffix for p in parts] == [f".part{i:02d}" for i in range(1, len(parts) + 1)]
digest = hashlib.sha256()
for part in parts:
    with part.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
(root / "sar-vlm-qwen3vl4b-arm64-cu126.tar.gz.sha256").write_text(
    digest.hexdigest() + "  sar-vlm-qwen3vl4b-arm64-cu126.tar.gz\n"
)
(root / "BUILD-INFO.txt").write_text(
    f"Platform: linux/arm64\nParts: {len(parts)}\nBytes: {sum(p.stat().st_size for p in parts)}\n"
    f"SHA256: {digest.hexdigest()}\nGPU forward/training: NOT RUN\n"
)
