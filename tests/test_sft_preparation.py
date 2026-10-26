"""Engineering fixtures exercise data conversion and split leakage checks only."""

import csv
import json

import pytest

from sar_vlm.cli.prepare_sft import prepare
from sar_vlm.training.offline_smoke import reviewed_subset


def labels(tmp_path, image_path, splits):
    path = tmp_path / "labels.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["image", "label", "source_group_id", "split", "annotation_status"]
        )
        writer.writeheader()
        for split in splits:
            writer.writerow(
                dict(
                    image=str(image_path),
                    label="ENGINEERING_FIXTURE",
                    source_group_id="fixture",
                    split=split,
                    annotation_status="verified",
                )
            )
    return path


def test_csv_to_train_is_readable(tmp_path, image_path):
    path = labels(tmp_path, image_path, ["train"])
    output = tmp_path / "sft"
    result = prepare(path, tmp_path, output, "unit test", "engineering fixture", "test only")
    assert result["counts"] == {"train": 1, "val": 0, "test": 0}
    rows = reviewed_subset(output / "train.jsonl", output / "train.review.json", None)
    assert len(rows) == 1
    assert rows[0]["images"] == [str(image_path)]
    assert json.loads((output / "data_manifest.json").read_text())["records"]


def test_same_target_cannot_leak_across_splits(tmp_path, image_path):
    path = labels(tmp_path, image_path, ["train", "test"])
    with pytest.raises(ValueError, match="leaks"):
        prepare(path, tmp_path, tmp_path / "sft", "test", "fixture", "test only")
    assert not (tmp_path / "sft").exists()


def test_unverified_label_rejected(tmp_path, image_path):
    path = labels(tmp_path, image_path, ["train"])
    path.write_text(path.read_text().replace("verified", "pending"))
    with pytest.raises(ValueError, match="verified"):
        prepare(path, tmp_path, tmp_path / "sft", "test", "fixture", "test only")
