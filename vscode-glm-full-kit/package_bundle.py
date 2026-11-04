"""Assemble a complete offline folder, including all already verified binaries."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
manifest = json.loads((a.root / "vscode-glm-kit/downloads.json").read_text(encoding="utf-8"))
for f in manifest["files"]:
    h = hashlib.sha256()
    with (a.root / f["name"]).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    assert h.hexdigest() == f["sha256"], f["name"]
shutil.copyfile("vscode-glm-full-kit/START-HERE.zh-CN.md", a.root / "START-HERE.zh-CN.md")
guide = a.root / "vscode-glm-kit/README.md"
old = guide.read_text(encoding="utf-8")
start = old.index("## 1. Windows 安装与 SSH")
guide.write_text(
    "# 完整离线安装与配置说明\n\n所有二进制已经位于上级目录。"
    "请先阅读 START-HERE.zh-CN.md，无需运行联网下载脚本。\n\n" + old[start:],
    encoding="utf-8",
)
a.output.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(a.output, "w", ZIP_STORED) as archive:
    for f in sorted(a.root.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            archive.write(f, Path("SAR-VSCode-Offline") / f.relative_to(a.root))
with ZipFile(a.output) as archive:
    assert archive.testzip() is None
print("Offline bundle complete:", a.output, a.output.stat().st_size)
