"""Offline transfer integrity tests use tiny structural fixtures, never real model claims."""

import json

import pytest

from sar_vlm.deployment.artifacts import create_manifest, validate_model_layout, verify_manifest


@pytest.fixture
def structural_model(tmp_path):
    # These are explicitly fake bytes for transfer validation, not an inference fixture.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "text_config": {"hidden_size": 4096, "num_hidden_layers": 32},
            }
        )
    )
    for name, data in {
        "tokenizer_config.json": '{"chat_template":"fixture"}',
        "tokenizer.json": "{}",
        "preprocessor_config.json": "{}",
        "model.safetensors.index.json": '{"weight_map":{"x":"shard.safetensors"}}',
        "shard.safetensors": "FAKE ENGINEERING FILE",
    }.items():
        (tmp_path / name).write_text(data)
    return tmp_path


def test_missing_shard_fails_layout(structural_model):
    (structural_model / "shard.safetensors").unlink()
    with pytest.raises(ValueError, match="missing/empty"):
        validate_model_layout(structural_model)


def test_manifest_byte_integrity_and_added_files(structural_model):
    create_manifest(structural_model)
    assert verify_manifest(structural_model)["model_weights_loaded"] is False
    (structural_model / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="set differs"):
        verify_manifest(structural_model)
    (structural_model / "extra.txt").unlink()
    (structural_model / "shard.safetensors").write_text("changed bytes")
    with pytest.raises(ValueError):
        verify_manifest(structural_model)


def test_manifest_rejects_path_traversal(structural_model):
    create_manifest(structural_model)
    path = structural_model / "transfer_manifest.json"
    data = json.loads(path.read_text())
    data["files"][0]["path"] = "../outside"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="invalid relative"):
        verify_manifest(structural_model)


def test_standalone_requires_regular_files(structural_model, tmp_path_factory):
    outside = tmp_path_factory.mktemp("external") / "weights"
    outside.write_text("outside")
    (structural_model / "shard.safetensors").unlink()
    (structural_model / "shard.safetensors").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        create_manifest(structural_model)


def test_4b_manifest_and_wrong_size_detection(structural_model):
    config = structural_model / "config.json"
    data = json.loads(config.read_text())
    data["text_config"] = {"hidden_size": 2560, "num_hidden_layers": 36}
    config.write_text(json.dumps(data))
    assert create_manifest(structural_model)["kind"] == "qwen3vl4b"
    assert verify_manifest(structural_model)["status"] == "PASS"
    data["text_config"]["hidden_size"] = 123
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="architecture"):
        validate_model_layout(structural_model)


def test_quantized_model_rejected(structural_model):
    config = structural_model / "config.json"
    data = json.loads(config.read_text())
    data["quantization_config"] = {"quant_method": "fp8"}
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unquantized"):
        validate_model_layout(structural_model)
