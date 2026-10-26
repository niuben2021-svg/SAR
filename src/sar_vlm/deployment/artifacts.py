"""Content manifests for standalone model directories transferred into an air gap."""

import argparse
import json
from pathlib import Path, PurePosixPath

from sar_vlm.data.reader import sha256_file
from sar_vlm.models.architecture import model_size
from sar_vlm.runtime import write_json

MANIFEST = "transfer_manifest.json"


def checked_path(root: Path, relative: str) -> Path:
    """Resolve a manifest entry inside root, rejecting traversal and escaping symlinks."""
    key = PurePosixPath(relative)
    if key.is_absolute() or ".." in key.parts or not key.parts:
        raise ValueError(f"invalid relative artifact path: {relative}")
    path = root.joinpath(*key.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"artifact escapes standalone directory: {relative}")
    return path


def validate_model_layout(root: Path) -> None:
    """Require the dense Qwen3-VL-4B/8B config, tokenizer, processor, chat and all shards."""
    config = json.loads((root / "config.json").read_text())
    model_size(config)
    tokenizer = json.loads((root / "tokenizer_config.json").read_text())
    if not (
        (root / "tokenizer.json").is_file()
        or all((root / p).is_file() for p in ("vocab.json", "merges.txt"))
    ):
        raise ValueError("tokenizer files are incomplete")
    if not (root / "preprocessor_config.json").is_file():
        raise ValueError("preprocessor_config.json is missing")
    if not (
        tokenizer.get("chat_template")
        or any((root / name).is_file() for name in ("chat_template.jinja", "chat_template.json"))
    ):
        raise ValueError("chat template missing; image/text inference cannot be reproduced")
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weights = json.loads(index.read_text()).get("weight_map", {})
        if not weights:
            raise ValueError("weight index is empty")
        shards = set(weights.values())
    else:
        shards = {"model.safetensors"}
    for shard in shards:
        path = checked_path(root, shard)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"model weight shard missing/empty: {shard}")


def create_manifest(root: Path, model: bool = True) -> dict:
    """Hash every deliverable file; exclude downloader caches and manifest itself."""
    root = root.resolve(strict=True)
    if model:
        validate_model_layout(root)
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts) or relative.as_posix() == MANIFEST:
            continue
        if path.is_file():
            checked_path(root, relative.as_posix())
            # Standalone tar transfers must not depend on Hub-cache symlinks.
            if path.is_symlink():
                raise ValueError(f"materialize model symlinks before transfer: {relative}")
            files.append(
                {
                    "path": relative.as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not files:
        raise ValueError("artifact directory is empty")
    result = {
        "schema_version": "1",
        "kind": ("qwen3vl" + model_size(json.loads((root / "config.json").read_text())).lower())
        if model
        else "files",
        "files": files,
        "scope": "local transfer integrity; not independent upstream weight provenance",
    }
    write_json(root / MANIFEST, result)
    return result


def verify_manifest(root: Path, model: bool = True) -> dict:
    """Require current bytes to exactly match the local transfer manifest."""
    root = root.resolve(strict=True)
    manifest = json.loads((root / MANIFEST).read_text())
    seen = set()
    for item in manifest["files"]:
        if item["path"] in seen:
            raise ValueError("duplicate path in transfer manifest")
        seen.add(item["path"])
        path = checked_path(root, item["path"])
        if not path.is_file() or path.is_symlink() or path.stat().st_size != item["size"]:
            raise ValueError(f"artifact missing, linked or size changed: {item['path']}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"artifact SHA256 mismatch: {item['path']}")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and not any(x.startswith(".") for x in p.relative_to(root).parts)
        and p.relative_to(root).as_posix() != MANIFEST
    }
    if seen != actual or not seen:
        raise ValueError("artifact set differs from transfer manifest")
    if model:
        if (
            manifest.get("kind")
            != "qwen3vl" + model_size(json.loads((root / "config.json").read_text())).lower()
        ):
            raise ValueError("not a model transfer manifest")
        validate_model_layout(root)
    return {"status": "PASS", "verified_files": len(seen), "model_weights_loaded": False}


def main() -> None:
    """Create/verify transfer checksums without loading large model weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "verify"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.action == "create":
        result = create_manifest(args.directory)
        print(f"Hashed {len(result['files'])} model files")
    else:
        print(json.dumps(verify_manifest(args.directory)))


if __name__ == "__main__":
    main()
