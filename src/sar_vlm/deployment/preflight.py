"""Offline dependency/device/layout checks before real-weight inference."""

import argparse
import importlib.metadata
from pathlib import Path

from sar_vlm.deployment.artifacts import validate_model_layout
from sar_vlm.models.qwen_adapter import require_device
from sar_vlm.runtime import command, timestamp, write_json


def main() -> int:
    """Record actual GPU visibility and local layout; do not claim a model forward."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/models/qwen3vl4b"))
    parser.add_argument("--output", type=Path, default=Path("/outputs/reports/preflight.json"))
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()
    import torch

    report = {
        "timestamp": timestamp(),
        "inference_executed": False,
        "errors": [],
        "packages": {
            k: importlib.metadata.version(k)
            for k in ("torch", "torchvision", "transformers", "ms-swift", "qwen-vl-utils")
        },
        "cuda_runtime": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "nvidia_smi": command(["nvidia-smi"]),
        "topology": command(["nvidia-smi", "topo", "-m"]),
    }
    try:
        validate_model_layout(args.model)
        report["layout"] = "PASS"
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        report["model_class"] = Qwen3VLForConditionalGeneration.__name__
        report["processor_class"] = Qwen3VLProcessor.__name__
        if not args.no_gpu:
            require_device("cuda:0")
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["status"] = "BLOCKED" if report["errors"] else "PASS"
    report["gpu_checked"] = not args.no_gpu
    write_json(args.output, report)
    print(f"Preflight {report['status']}; CUDA GPUs={report['gpu_count']}; inference NOT RUN")
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
