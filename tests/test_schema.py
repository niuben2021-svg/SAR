"""Coordinate, provenance and canonical JSONL failure cases."""

import json

import pytest
from pydantic import ValidationError

from sar_vlm.data.schema import CanonicalSample, read_manifest


def parse(data):
    return CanonicalSample.model_validate_json(json.dumps(data))


def test_empty_objects_and_unknown_metadata(canonical_dict):
    sample = parse(canonical_dict)
    assert sample.objects == []
    assert sample.metadata.sensor is None
    assert sample.metadata.incidence_angle_deg is None


def test_multiclass_and_boundary(canonical_dict):
    canonical_dict["objects"] = [
        {
            "object_id": str(i),
            "bbox_xyxy": box,
            "coarse_class": name,
            "annotation_status": "synthetic_fixture",
        }
        for i, (box, name) in enumerate(
            [([0, 0, 64, 48], "aircraft"), ([2, 3, 4.5, 6.5], "vehicle")]
        )
    ]
    assert len(parse(canonical_dict).objects) == 2


@pytest.mark.parametrize(
    "box",
    [
        [-1, 0, 2, 2],
        [0, 0, 65, 2],
        [0, 0, 2, 49],
        [2, 2, 2, 4],
        [4, 2, 2, 4],
        [0, 0, float("nan"), 3],
        [0, 0, float("inf"), 3],
    ],
)
def test_invalid_boxes(canonical_dict, box):
    canonical_dict["objects"] = [
        {
            "object_id": "x",
            "bbox_xyxy": box,
            "coarse_class": "vehicle",
            "annotation_status": "pending",
        }
    ]
    with pytest.raises(ValidationError):
        parse(canonical_dict)


@pytest.mark.parametrize("key", ["source_group_id", "query", "label_provenance", "width"])
def test_missing_required_fields(canonical_dict, key):
    del canonical_dict[key]
    with pytest.raises(ValidationError):
        parse(canonical_dict)


def test_invalid_nested_evidence_and_history(canonical_dict):
    canonical_dict["objects"] = [
        {
            "object_id": "x",
            "bbox_xyxy": [0, 0, 2, 2],
            "coarse_class": "vehicle",
            "annotation_status": "pending",
            "evidence": [{"type": "geometry", "name": "invalid", "bbox_xyxy": [0, 0, 100, 1]}],
        }
    ]
    with pytest.raises(ValidationError, match="bbox_xyxy"):
        parse(canonical_dict)
    canonical_dict["objects"] = []
    canonical_dict["historical_priors"] = [{"label": "unvalidated"}]
    with pytest.raises(ValidationError, match="KNOW-01"):
        parse(canonical_dict)


def test_manifest_context(manifest_path):
    assert len(read_manifest(manifest_path)) == 1
    manifest_path.write_text('\n{"invalid":true}\n')
    with pytest.raises(ValueError, match=r":2:"):
        read_manifest(manifest_path)
    manifest_path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_manifest(manifest_path)


@pytest.mark.parametrize("mutation", [{"split": "train"}, {"label_provenance": "verified"}])
def test_synthetic_provenance_cannot_be_promoted(canonical_dict, mutation):
    canonical_dict.update(mutation)
    with pytest.raises(ValidationError, match="fixture split/provenance"):
        parse(canonical_dict)
