"""Ensure training cannot consume unreviewed bytes or an unrelated forward result."""

import json

import pytest

from sar_vlm.data.reader import sha256_file
from sar_vlm.training.offline_smoke import require_forward, reviewed_subset


def test_reviewed_subset_binds_exact_data_bytes(tmp_path, image_path):
    dataset = tmp_path / "sft.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": "<image>工程接口检查"},
            {"role": "assistant", "content": "此记录仅用于测试数据约束。"},
        ],
        "images": [str(image_path)],
    }
    dataset.write_text(json.dumps(row) + "\n")
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "annotation_status": "verified",
                "source_dataset": "TEST_ONLY",
                "reviewed_by": "test",
                "license": "engineering fixture",
                "sft_sha256": sha256_file(dataset),
            }
        )
    )
    subset = reviewed_subset(dataset, review, 100)
    assert len(subset) == 1
    dataset.write_text(dataset.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        reviewed_subset(dataset, review, 100)


def test_failed_forward_cannot_enable_training(tmp_path):
    report = tmp_path / "forward.json"
    report.write_text('{"status":"BLOCKED","inference_executed":false}')
    with pytest.raises(ValueError, match="successful real"):
        require_forward(report, tmp_path)


def test_full_training_requires_completed_single_gpu_smoke(tmp_path):
    from sar_vlm.training.offline_smoke import require_smoke

    dataset = tmp_path / "train.jsonl"
    dataset.write_text("TEST_ONLY")
    manifest = tmp_path / "transfer_manifest.json"
    manifest.write_text("TEST_ONLY")
    proof = {
        "status": "PASS",
        "training_executed": True,
        "processes": 1,
        "mode": "smoke",
        "completed_steps": 3,
        "dataset_sha256": sha256_file(dataset),
        "weights_manifest_sha256": sha256_file(manifest),
    }
    report = tmp_path / "run.json"
    report.write_text(json.dumps(proof))
    require_smoke(report, dataset, tmp_path)
    dataset.write_text("DIFFERENT_DATA")
    with pytest.raises(ValueError, match="single-GPU smoke"):
        require_smoke(report, dataset, tmp_path)
