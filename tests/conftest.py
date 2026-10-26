"""Synthetic engineering fixtures only; never training/evaluation ground truth."""

import json

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "fixture.png"
    Image.fromarray(np.arange(64 * 48, dtype=np.uint16).reshape(48, 64).astype(np.uint8)).save(path)
    return path


@pytest.fixture
def canonical_dict():
    return {
        "schema_version": "0.1",
        "scene_id": "engineering_fixture_1",
        "source_image": "fixture.png",
        "width": 64,
        "height": 48,
        "image_type": "SYNTHETIC_FIXTURE",
        "source_group_id": "fixture_group",
        "split": "fixture",
        "query": {"text": "工程检查", "target_classes": []},
        "objects": [],
        "label_provenance": "synthetic_fixture",
    }


@pytest.fixture
def manifest_path(image_path, canonical_dict):
    path = image_path.with_suffix(".jsonl")
    path.write_text(json.dumps(canonical_dict) + "\n", encoding="utf-8")
    return path
