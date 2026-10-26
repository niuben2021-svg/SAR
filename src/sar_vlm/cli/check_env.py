"""Record actual hardware, dependencies, paths and missing prerequisites."""

import argparse
import importlib.metadata
import os
import platform
import shutil
from pathlib import Path

from sar_vlm.config import load_config
from sar_vlm.runtime import command, config_hash, timestamp, write_json


def main() -> int:
    """Collect diagnostics without reading secret values or downloading models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    versions = {}
    for package in (
        "torch",
        "torchvision",
        "transformers",
        "ms-swift",
        "numpy",
        "Pillow",
        "pydantic",
        "PyYAML",
        "pytest",
        "ruff",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    gpu = {"count": None, "cuda_runtime": None, "devices": [], "error": None}
    try:
        import torch

        gpu.update(
            count=torch.cuda.device_count(),
            cuda_runtime=torch.version.cuda,
            devices=[
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                    "free_memory_bytes": torch.cuda.mem_get_info(i)[0],
                }
                for i in range(torch.cuda.device_count())
            ],
        )
    except ImportError:
        gpu["error"] = "torch not installed; device count cannot be queried through PyTorch"
    caches = [Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))]
    if "HF_HOME" in os.environ:
        caches.append(Path(os.environ["HF_HOME"]) / "hub")
    found = sorted(
        {
            str(p)
            for root in caches
            if root.is_dir()
            for p in root.glob("models--Qwen--Qwen3-VL-*B-Instruct/snapshots/*")
        }
    )
    report = {
        "timestamp": timestamp(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "gpu": gpu,
        "git": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "nvidia_smi": command(["nvidia-smi"]),
        "nvidia_topology": command(["nvidia-smi", "topo", "-m"]),
        "disk_free_bytes": shutil.disk_usage(args.config.resolve().parent).free,
        "config_hash": config_hash(args.config),
        "model_path": cfg.model_path,
        "requested_revision": cfg.revision,
        "local_model_exists": Path(cfg.model_path).is_dir(),
        "qwen_cache_snapshots": found,
        "model_cache_search_scope": [str(root) for root in caches],
        "weights_verified": False,
        "input_image": str(cfg.input_image) if cfg.input_image else None,
        "manifest": str(cfg.manifest) if cfg.manifest else None,
        "verified_sar_input_configured": False,
        "notes": [
            "Cache directory presence does not prove complete or verified weights.",
            "Configured input provenance must be checked via smoke_data.",
            "CPU package versions do not validate a future CUDA installation.",
        ],
    }
    write_json(cfg.report_dir / "environment.json", report)
    (cfg.report_dir / "environment.md").write_text(
        "# Actual environment\n\n```json\n"
        + (cfg.report_dir / "environment.json").read_text()
        + "```\n",
        encoding="utf-8",
    )
    print(f"Environment recorded: {cfg.report_dir / 'environment.json'}; CUDA count={gpu['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
