"""Download the complete official Qwen snapshot on an Internet-connected machine."""

import argparse
from pathlib import Path

from sar_vlm.deployment.artifacts import create_manifest
from sar_vlm.models.architecture import DEFAULT_MODEL_ID
from sar_vlm.runtime import timestamp, write_json


def main() -> None:
    """Download all files through a selected official provider, then hash for transfer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface")
    parser.add_argument(
        "--revision", help="Provider-specific commit/tag; main/master resolved online"
    )
    parser.add_argument(
        "--model-id",
        choices=[DEFAULT_MODEL_ID, "Qwen/Qwen3-VL-8B-Instruct"],
        default=DEFAULT_MODEL_ID,
    )
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "download_source.json"
    # Preserve an unrelated pre-existing model directory rather than merging snapshots.
    if any(destination.iterdir()) and not marker.is_file():
        parser.error("destination must be empty or previously initialized by this downloader")
    requested = args.revision or ("main" if args.source == "huggingface" else "master")
    provenance = {
        "model_id": args.model_id,
        "provider": args.source,
        "requested_revision": requested,
        "status": "DOWNLOAD_INCOMPLETE",
        "timestamp": timestamp(),
    }
    if marker.is_file():
        import json

        previous = json.loads(marker.read_text())
        if any(
            previous.get(k) != provenance[k] for k in ("provider", "model_id", "requested_revision")
        ):
            parser.error("existing directory belongs to a different source/revision; use a new one")
    write_json(marker, provenance)
    if args.source == "huggingface":
        from huggingface_hub import HfApi, snapshot_download

        revision = HfApi().model_info(args.model_id, revision=requested).sha
        snapshot_download(repo_id=args.model_id, revision=revision, local_dir=destination)
        provenance["resolved_revision"] = revision
    else:
        from modelscope import snapshot_download

        snapshot_download(args.model_id, revision=requested, local_dir=str(destination))
        provenance["resolved_revision"] = None
        provenance["revision_note"] = (
            "Provider revision requested; transfer hashes identify exact bytes."
        )
    provenance.update(status="DOWNLOAD_COMPLETE", completed_at=timestamp())
    write_json(marker, provenance)
    result = create_manifest(destination)
    print(f"Downloaded and hashed {len(result['files'])} files to {destination}")


if __name__ == "__main__":
    main()
