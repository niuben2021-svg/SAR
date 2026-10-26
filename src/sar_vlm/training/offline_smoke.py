"""Explicit, bounded LoRA smoke launcher for a reviewed local SFT subset."""

import argparse
import json
import os
import subprocess
import uuid
from pathlib import Path

from sar_vlm.data.reader import SARReader, sha256_file
from sar_vlm.deployment.artifacts import validate_model_layout
from sar_vlm.models.qwen_adapter import require_device
from sar_vlm.runtime import command, timestamp, write_json
from sar_vlm.training.config import load_training_config


def reviewed_subset(dataset: Path, review: Path, limit: int | None) -> list[dict]:
    """Validate human review binding and real local image paths; never infer labels."""
    if limit is not None and not 1 <= limit <= 500:
        raise ValueError("smoke subset limit must be 1..500")
    attestation = json.loads(review.read_text())
    if attestation.get("annotation_status") != "verified" or any(
        not attestation.get(key) for key in ("source_dataset", "license", "reviewed_by")
    ):
        raise ValueError("review requires verified status, dataset, license and reviewed_by")
    if attestation.get("sft_sha256") != sha256_file(dataset):
        raise ValueError("review hash does not match the SFT JSONL bytes")
    samples = []
    with dataset.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages, images = row.get("messages"), row.get("images")
            if (
                not isinstance(messages, list)
                or len(messages) < 2
                or not isinstance(images, list)
                or not images
            ):
                raise ValueError(f"line {number}: require messages and nonempty images")
            if any(
                not isinstance(m, dict)
                or m.get("role") not in {"system", "user", "assistant"}
                or not isinstance(m.get("content"), str)
                or not m["content"].strip()
                for m in messages
            ):
                raise ValueError(f"line {number}: invalid message roles/content")
            if not any(m["role"] == "assistant" for m in messages):
                raise ValueError(f"line {number}: no reviewed assistant answer")
            if sum(m["content"].count("<image>") for m in messages) != len(images):
                raise ValueError(f"line {number}: <image> count differs from images")
            if not isinstance(images, list) or not all(isinstance(p, str) for p in images):
                raise ValueError(f"line {number}: images must be a list of local paths")
            resolved = [(dataset.parent / p).resolve(strict=True) for p in images]
            for path in resolved:
                record = SARReader().read(path)
                if record.dtype != "uint8":
                    raise ValueError("SFT model images require explicit 8-bit display preparation")
            row["images"] = [str(p) for p in resolved]
            if row.get("annotation_status", "verified") != "verified":
                raise ValueError(f"line {number}: non-verified training record")
            samples.append({k: v for k, v in row.items() if k in {"messages", "images", "objects"}})
            if len(samples) == limit:
                break
    if not samples:
        raise ValueError("reviewed dataset is empty")
    return samples


def require_forward(report: Path, model: Path) -> None:
    """Require a successful real GPU forward for the exact mounted model artifact."""
    proof = json.loads(report.read_text())
    if proof.get("status") != "PASS" or proof.get("inference_executed") is not True:
        raise ValueError("a successful real single-GPU forward is required before training")
    if (
        not proof.get("gpu_peak_memory_bytes")
        or Path(proof.get("weights", "")).resolve() != model.resolve()
    ):
        raise ValueError("forward report device/model does not match this training run")
    manifest = model / "transfer_manifest.json"
    if not manifest.is_file() or proof.get("weights_manifest_sha256") != sha256_file(manifest):
        raise ValueError("forward proof must bind to this model's transfer_manifest.json")


def require_smoke(report: Path, dataset: Path, model: Path) -> None:
    """Bind full training to a completed single-GPU smoke on the same model and data."""
    proof = json.loads(report.read_text())
    if (
        proof.get("status") != "PASS"
        or proof.get("training_executed") is not True
        or proof.get("processes") != 1
        or proof.get("mode") != "smoke"
        or proof.get("completed_steps", 0) < 1
        or proof.get("dataset_sha256") != sha256_file(dataset)
        or proof.get("weights_manifest_sha256") != sha256_file(model / "transfer_manifest.json")
    ):
        raise ValueError("completed single-GPU smoke required for the same model and dataset")


def main(full: bool = False) -> int:
    """Plan by default; execution requires real GPU/forward checks and an explicit budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/models/qwen3vl4b"))
    parser.add_argument("--config", type=Path, default=Path("configs/qwen3vl4b_lora_2xa100.yaml"))
    parser.add_argument(
        "--forward-report", type=Path, default=Path("/outputs/mvp/forward_output.json")
    )
    parser.add_argument("--output", type=Path, default=Path("/outputs/training"))
    parser.add_argument("--processes", type=int, choices=[1, 2], default=1)
    parser.add_argument("--steps", type=int, default=100 if full else 3)
    if not full:
        parser.add_argument("--limit", type=int, default=100)
    else:
        parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument(
        "--execute", action="store_true", help="actually run the requested training step budget"
    )
    args = parser.parse_args()
    if not 1 <= args.steps <= (1000000 if full else 10):
        parser.error("invalid step budget (smoke: 1..10; train: 1..1000000)")
    validate_model_layout(args.model)
    rows = reviewed_subset(args.dataset, args.review, None if full else args.limit)
    cfg = load_training_config(args.config)
    run_dir = args.output / (("train-" if full else "smoke-") + uuid.uuid4().hex[:12])
    run_dir.mkdir(parents=True)
    subset = run_dir / "reviewed_subset.jsonl"
    subset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    tokens = []
    for key, value in cfg.model_dump().items():
        if key not in {"nproc_per_node", "image_max_token_num", "model"}:
            tokens.extend(
                [f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)]
            )
    tokens += [
        "--model",
        str(args.model.resolve()),
        "--dataset",
        str(subset.resolve()),
        "--output_dir",
        str((run_dir / "checkpoints").resolve()),
        "--max_steps",
        str(args.steps),
        "--split_dataset_ratio",
        "0",
        "--save_strategy",
        "steps",
        "--save_steps",
        str(args.steps),
        "--save_total_limit",
        "1",
        "--logging_steps",
        "1",
        "--dataset_num_proc",
        "1",
        "--dataloader_num_workers",
        "0",
        "--report_to",
        "none",
        "--add_version",
        "false",
    ]
    # Installed official parser, without dataclass post-init/model initialization.
    from swift.arguments import SftArguments
    from transformers import HfArgumentParser

    HfArgumentParser(SftArguments).parse_args(tokens)
    report = {
        "timestamp": timestamp(),
        "mode": "train" if full else "smoke",
        "status": "PLAN_VALIDATED",
        "settings": cfg.model_dump(),
        "model": str(args.model.resolve()),
        "weights_manifest_sha256": sha256_file(args.model / "transfer_manifest.json"),
        "software": command(["python", "-m", "pip", "freeze"]),
        "hardware": command(["nvidia-smi"]),
        "git": command(["git", "rev-parse", "HEAD"]),
        "training_executed": False,
        "command": ["swift", "sft", *tokens],
        "max_steps": args.steps,
        "processes": args.processes,
        "samples": len(rows),
        "seed": cfg.seed,
        "dataset_sha256": sha256_file(args.dataset),
        "review_sha256": sha256_file(args.review),
        "config_sha256": sha256_file(args.config),
        "run_directory": str(run_dir),
    }
    write_json(run_dir / "run.json", report)
    if not args.execute:
        print(f"Plan validated: {run_dir / 'run.json'}; TRAIN_NOT_EXECUTED")
        return 0
    try:
        require_forward(args.forward_report, args.model)
        if full:
            require_smoke(args.smoke_report, args.dataset, args.model)
        require_device("cuda:0")
        import torch

        if torch.cuda.device_count() < args.processes:
            raise ValueError("insufficient visible GPU count")
    except (OSError, ValueError, RuntimeError) as exc:
        report.update(status="BLOCKED", error=str(exc))
        write_json(run_dir / "run.json", report)
        print(f"Training BLOCKED: {exc}")
        return 2
    environment = {
        **os.environ,
        "NPROC_PER_NODE": str(args.processes),
        "IMAGE_MAX_TOKEN_NUM": str(cfg.image_max_token_num),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
    }
    log = run_dir / "train.log"
    report.update(status="RUNNING", training_executed=True)
    write_json(run_dir / "run.json", report)
    with log.open("w") as stream:
        result = subprocess.run(
            ["swift", "sft", *tokens], env=environment, stdout=stream, stderr=subprocess.STDOUT
        )
    states = list((run_dir / "checkpoints").rglob("trainer_state.json"))
    completed = max((json.loads(p.read_text()).get("global_step", 0) for p in states), default=0)
    passed = result.returncode == 0 and completed >= args.steps
    report.update(
        status="PASS" if passed else "FAILED",
        exit_code=result.returncode,
        completed_steps=completed,
        log=str(log),
        gpu_peak_memory_bytes=None,
    )
    write_json(run_dir / "run.json", report)
    print(f"Training {report['status']}: {log}; measured GPU peak not collected")
    return 0 if passed else (result.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
