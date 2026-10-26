"""Configuration resolution and real CLI integration checks."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sar_vlm.config import load_config
from sar_vlm.training.config import check_swift_flags, load_training_config

ROOT = Path(__file__).resolve().parents[1]


def test_config_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(ROOT / "configs/mvp.yaml")
    assert cfg.output_dir == ROOT / "outputs/mvp"
    assert cfg.device == "cuda:0"


@pytest.mark.parametrize("value", [0, -1])
def test_bad_budget(tmp_path, value):
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump({"output_dir": "out", "report_dir": "report", "image_max_tokens": value})
    )
    with pytest.raises(ValidationError):
        load_config(path)


def test_missing_input(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("output_dir: out\nreport_dir: report\ninput_image: missing.png\n")
    with pytest.raises(FileNotFoundError, match="input_image"):
        load_config(path)


def test_training_starting_values_and_exact_flags():
    cfg = load_training_config(ROOT / "configs/qwen3vl_lora_4xa100.yaml")
    assert (
        cfg.nproc_per_node * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps == 16
    )
    assert cfg.lora_rank == 16 and cfg.freeze_vit and cfg.freeze_aligner
    assert check_swift_flags(
        "--model_type TYPE --max_length LENGTH", ["--model", "--max_length"]
    ) == ["--model"]


def test_data_cli_real_io_and_repeatability(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("output_dir: out\nreport_dir: reports\n")
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-m", "sar_vlm.cli.smoke_data", "--config", str(config)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        trace = json.loads((tmp_path / "out/data_trace.json").read_text())
        assert trace["samples"][0]["label_provenance"] == "synthetic_fixture"
        assert trace["samples"][0]["dtype"] == "uint8"
        assert not trace["inference_executed"]


def test_forward_failure_is_nonzero_and_not_fake_success(tmp_path, image_path):
    config = tmp_path / "config.yaml"
    # Force an unavailable index on any normal machine, without touching model caches.
    config.write_text("output_dir: out\nreport_dir: reports\ndevice: cuda:99999\n")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sar_vlm.cli.smoke_forward",
            "--config",
            str(config),
            "--image",
            str(image_path),
            "--query",
            "工程检查",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    trace = json.loads((tmp_path / "out/forward_failure.json").read_text())
    assert trace["status"] == "BLOCKED" and not trace["inference_executed"]
    assert "raw_text" not in trace
    assert not (tmp_path / "out/forward_output.json").exists()


def test_training_missing_cli_is_honest(tmp_path):
    env = {**os.environ, "PATH": str(tmp_path)}
    report = tmp_path / "validation.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sar_vlm.training.config",
            "--config",
            str(ROOT / "configs/qwen3vl_lora_4xa100.yaml"),
            "--report",
            str(report),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    data = json.loads(report.read_text())
    assert data["schema_status"] == "PASS"
    assert data["cli_status"] == "BLOCKED"
    assert data["training_status"] == "TRAIN_NOT_EXECUTED"


def test_fixture_generation_protects_modified_manifest(tmp_path):
    from sar_vlm.cli.smoke_data import make_fixture

    manifest = make_fixture(tmp_path)
    data = json.loads(manifest.read_text())
    data["scene_id"] = "user_changed"
    manifest.write_text(json.dumps(data) + "\n")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        make_fixture(tmp_path)
    assert json.loads(manifest.read_text())["scene_id"] == "user_changed"


def test_confirmed_two_a100_config():
    cfg = load_training_config(ROOT / "configs/qwen3vl_lora_2xa100.yaml")
    assert cfg.nproc_per_node == 2
    assert cfg.gradient_accumulation_steps == 8
    assert (
        cfg.nproc_per_node * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps == 16
    )
