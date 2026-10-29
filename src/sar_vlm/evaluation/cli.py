"""Prepare, infer and score held-out SAR data without any training or network access."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from sar_vlm.config import MVPConfig
from sar_vlm.data.reader import SARReader, sha256_file
from sar_vlm.evaluation.prepare import prepare
from sar_vlm.evaluation.runner import evaluate, parse_output, prompt, write_reports
from sar_vlm.evaluation.schema import (
    Prediction,
    Protocol,
    json_line,
    read_predictions,
    read_samples,
)
from sar_vlm.models.qwen_adapter import QwenAdapter
from sar_vlm.runtime import command, timestamp, write_json


def check_inputs(
    samples: list, manifest: Path, image_root: Path | None, exclude_manifest: Path | None
) -> dict[str, Path]:
    """Check all image hashes/sizes before loading weights; optionally audit training leakage."""
    excluded_groups, excluded_hashes = set(), set()
    if exclude_manifest is not None:
        # Accept the data_manifest.json emitted by prepare-sft, excluding held-out records.
        data = json.loads(exclude_manifest.read_text())
        for row in data["records"]:
            if row["split"] == "train":
                excluded_groups.add(row["source_group_id"])
                excluded_hashes.add(row["image_sha256"])
    paths = {}
    root = image_root.resolve(strict=True) if image_root else manifest.parent.resolve()
    for sample in samples:
        path = (root / sample.image).resolve(strict=True)
        if image_root and not path.is_relative_to(root):
            raise ValueError(f"{sample.id}: image escapes image-root")
        record = SARReader().read(path)
        if (record.width, record.height) != (sample.width, sample.height):
            raise ValueError(f"{sample.id}: manifest/image size mismatch")
        if record.dtype != "uint8":
            raise ValueError("convert a separate uint8 display copy before inference")
        if sample.image_sha256 and record.sha256 != sample.image_sha256:
            raise ValueError(f"{sample.id}: image hash mismatch")
        if sample.source_group_id in excluded_groups or record.sha256 in excluded_hashes:
            raise ValueError(f"{sample.id}: overlaps training group or image hash")
        paths[sample.id] = path
    return paths


def checkpoint_hashes(path: Path | None) -> dict:
    """Hash LoRA parameters/config for reproducibility; do not infer dataset provenance."""
    if path is None:
        return {}
    path = path.resolve(strict=True)
    files = [
        path / "adapter_config.json",
        *path.glob("adapter_model*.safetensors"),
        *path.glob("adapter_model*.bin"),
    ]
    if len(files) < 2:
        raise ValueError("adapter directory requires adapter_config.json and adapter weights")
    return {p.name: sha256_file(p) for p in files}


def run(args: argparse.Namespace) -> int:
    """Execute one CLI operation; model operations are only in the infer branch."""
    if args.operation == "prepare":
        print(f"Prepared {prepare(args.csv, args.image_root, args.output, args.split)} samples")
        return 0
    protocol = Protocol.model_validate_json(args.protocol.read_text())
    samples = read_samples(args.manifest, protocol, args.allow_fixtures)
    if args.output.exists():
        raise ValueError("output directory exists; use a new directory to preserve earlier runs")
    args.output.mkdir(parents=True)
    write_json(args.output / "protocol.json", protocol.model_dump())
    base = {
        "timestamp": timestamp(),
        "manifest_sha256": sha256_file(args.manifest),
        "protocol_sha256": sha256_file(args.protocol),
        "git": command(["git", "rev-parse", "HEAD"]),
        "source_sha256": {
            str(p.relative_to(Path(__file__).parents[1])): sha256_file(p)
            for p in sorted(Path(__file__).parents[1].rglob("*.py"))
        },
        "inference_executed": False,
        "training_executed": False,
    }
    if args.operation == "score":
        predictions = read_predictions(args.predictions)
        base["predictions_sha256"] = sha256_file(args.predictions)
    else:
        try:
            paths = check_inputs(samples, args.manifest, args.image_root, args.exclude_manifest)
            model_path = args.model.resolve(strict=True)
            cfg = MVPConfig(
                model_path=str(model_path),
                device=args.device,
                local_files_only=True,
                output_dir=args.output,
                report_dir=args.output,
                max_new_tokens=args.max_new_tokens,
                image_max_tokens=args.image_tokens,
                seed=args.seed,
            )
            base.update(
                model=str(model_path),
                adapter=str(args.adapter) if args.adapter else None,
                adapter_hashes=checkpoint_hashes(args.adapter),
                exclude_manifest_sha256=sha256_file(args.exclude_manifest)
                if args.exclude_manifest
                else None,
                training_overlap_audit="CHECKED" if args.exclude_manifest else "NOT_CHECKED",
                settings=cfg.model_dump(mode="json"),
                versions={
                    k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft")
                },
                nvidia_smi=command(["nvidia-smi"]),
                nvidia_topology=command(["nvidia-smi", "topo", "-m"]),
            )
            shared_prompt = prompt(args.task, protocol)
            (args.output / "prompt.txt").write_text(shared_prompt + "\n")
            model = QwenAdapter(cfg, adapter_path=args.adapter)
            model.load()
        except Exception as exc:
            write_json(
                args.output / "run.json",
                {**base, "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        predictions = []
        start = time.perf_counter()
        with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for sample in samples:
                try:
                    trace = model.generate(paths[sample.id], shared_prompt)
                except Exception as exc:
                    pred = Prediction(
                        id=sample.id, status="inference_error", error=f"{type(exc).__name__}: {exc}"
                    )
                else:
                    try:
                        pred = parse_output(trace["raw_text"], sample, args.task, protocol)
                    except (ValueError, TypeError, KeyError) as exc:
                        pred = Prediction(
                            id=sample.id,
                            status="parse_error",
                            raw_text=trace["raw_text"],
                            error=str(exc),
                        )
                    pred.trace = trace
                stream.write(json_line(pred))
                stream.flush()
                predictions.append(pred)
                print(f"{len(predictions)}/{len(samples)} {sample.id}: {pred.status}", flush=True)
        base.update(
            inference_executed=True,
            elapsed_seconds=time.perf_counter() - start,
            load_seconds=model.load_seconds,
            image_protocol="whole-image resized generation baseline; no tiling/Object Head",
        )
    report = evaluate(samples, predictions, protocol, args.task)
    by_id = {p.id: p for p in predictions}
    with (args.output / "comparisons.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            prediction = by_id.get(sample.id, Prediction(id=sample.id, status="missing"))
            stream.write(
                json.dumps(
                    {
                        "reference": sample.model_dump(mode="json"),
                        "prediction": prediction.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report["provenance"] = base
    write_reports(args.output, report)
    write_json(args.output / "run.json", {**base, "status": report["status"]})
    print(f"{report['status']}: {args.output / 'report.md'}")
    return 1 if report["failed_samples"] else 0


def main() -> int:
    """Provide CPU-only score/prepare and offline base-model or LoRA inference modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prep = sub.add_parser("prepare", help="convert reviewed training-compatible CSV")
    prep.add_argument("--csv", type=Path, required=True)
    prep.add_argument("--image-root", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--split", choices=["test", "val"], default="test")
    for name in ("infer", "score"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--protocol", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument(
            "--task",
            choices=["classification", "detection", "description", "all"],
            default="classification",
        )
        p.add_argument("--allow-fixtures", action="store_true")
        if name == "score":
            p.add_argument("--predictions", type=Path, required=True)
        else:
            p.add_argument("--model", type=Path, default=Path("/models/qwen3vl4b"))
            p.add_argument("--adapter", type=Path)
            p.add_argument("--image-root", type=Path)
            p.add_argument(
                "--exclude-manifest",
                type=Path,
                help="prepare-sft data_manifest.json for training leakage audit",
            )
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--image-tokens", type=int, default=1024)
            p.add_argument("--max-new-tokens", type=int, default=1024)
            p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
