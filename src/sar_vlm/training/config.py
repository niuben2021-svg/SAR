"""Validate LoRA starting settings against the actual installed ms-swift CLI."""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sar_vlm.runtime import config_hash, timestamp, write_json


class TrainingConfig(BaseModel):
    """Typed initial LoRA settings; no training starts during validation."""

    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    nproc_per_node: Literal[1, 2, 4]
    image_max_token_num: int = Field(gt=0)
    seed: int = Field(ge=0)
    tuner_type: Literal["lora"]
    torch_dtype: Literal["bfloat16"]
    lora_rank: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    target_modules: Literal["all-linear"]
    freeze_vit: bool
    freeze_aligner: bool
    per_device_train_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    max_length: int = Field(gt=0)
    gradient_checkpointing: bool
    attn_impl: Literal["sdpa"]
    num_train_epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)


def load_training_config(path: Path) -> TrainingConfig:
    """Validate YAML fields/types; this is not runtime compatibility validation."""
    return TrainingConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def check_swift_flags(help_text: str, flags: list[str]) -> list[str]:
    """Return flags absent from actual help output, matching complete flag names."""
    supported = set(re.findall(r"--[A-Za-z][A-Za-z0-9_\-]*", help_text))
    return [flag for flag in flags if flag not in supported]


def main() -> int:
    """Check settings/installed CLI without starting any training or downloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("reports/runs/config_validation.json"))
    parser.add_argument(
        "--dry-run", action="store_true", help="validation only; always no training"
    )
    args = parser.parse_args()
    cfg = load_training_config(args.config)
    flags = [
        f"--{key}"
        for key in cfg.model_dump()
        if key not in {"nproc_per_node", "image_max_token_num"}
    ]
    report = {
        "timestamp": timestamp(),
        "config_hash": config_hash(args.config),
        "schema_status": "PASS",
        "training_status": "TRAIN_NOT_EXECUTED",
        "settings": cfg.model_dump(),
        "cli_status": "BLOCKED",
        "required_flags": flags,
        "launch_environment": {
            "NPROC_PER_NODE": cfg.nproc_per_node,
            "IMAGE_MAX_TOKEN_NUM": cfg.image_max_token_num,
        },
        "notes": [
            "Requires verified SFT data and single-GPU forward before training.",
            "No dataset paths or training results are fabricated.",
        ],
    }
    executable = shutil.which("swift")
    if executable is None:
        report["reason"] = "ms-swift CLI not installed in the selected environment"
        code = 2
    else:
        try:
            proc = subprocess.run(
                [executable, "sft", "--help"], capture_output=True, text=True, timeout=60
            )
            report.update(
                help_exit_code=proc.returncode, help_stdout=proc.stdout, help_stderr=proc.stderr
            )
            missing = check_swift_flags(proc.stdout, flags)
            report["unsupported_flags"] = missing
            passed = proc.returncode == 0 and not missing
            report["cli_status"] = "PASS" if passed else "FAILED"
            code = 0 if passed else 1
            if proc.returncode == 0 and missing:
                # ms-swift 4.6.0's preliminary argparse parser intercepts --help.
                # Retain its actual result and separately validate official SftArguments.
                report["cli_status"] = "INCOMPLETE_HELP"
                parser_report = args.report.with_name("swift_parser_validation.json")
                fallback = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "sar_vlm.training.swift_parser",
                        "--config",
                        str(args.config),
                        "--report",
                        str(parser_report),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                report["official_parser"] = {
                    "exit_code": fallback.returncode,
                    "status": "PASS" if fallback.returncode == 0 else "FAILED",
                    "stdout": fallback.stdout,
                    "stderr": fallback.stderr,
                    "report": str(parser_report),
                }
                code = 0 if fallback.returncode == 0 else 1
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["reason"] = str(exc)
            code = 2
    write_json(args.report, report)
    parser_status = report.get("official_parser", {}).get("status", "NOT_RUN")
    print(
        f"YAML=PASS; swift CLI={report['cli_status']}; "
        f"official parser={parser_status}; TRAIN_NOT_EXECUTED"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
