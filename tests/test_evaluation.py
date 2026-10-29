"""Synthetic engineering checks, never empirical SAR accuracy measurements."""

import csv
import json
import subprocess
import sys

import pytest
from PIL import Image

from sar_vlm.data.conversion import qwen1000_to_pixels
from sar_vlm.evaluation.cli import check_inputs
from sar_vlm.evaluation.metrics import ap101, iou, text_pair
from sar_vlm.evaluation.prepare import prepare
from sar_vlm.evaluation.runner import evaluate, parse_output, prompt
from sar_vlm.evaluation.schema import (
    Detection,
    Prediction,
    Protocol,
    Sample,
    json_line,
    read_samples,
)


@pytest.fixture
def protocol():
    return Protocol(classes=["飞机", "车辆"], attribute_values={"阴影": ["明显", "不明显"]})


def sample(id="a", **kwargs):
    return Sample(
        id=id,
        image="image.png",
        width=100,
        height=100,
        source_group_id=id,
        split="fixture",
        annotation_status="synthetic_fixture",
        **kwargs,
    )


def box(score=0.9, label="飞机", xy=(0.0, 0.0, 20.0, 20.0)):
    return Detection(label=label, bbox_xyxy=xy, raw_score=score)


def test_half_open_geometry_and_conversion():
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert qwen1000_to_pixels((0, 0, 500, 500), 200, 80) == (0, 0, 100, 40)
    with pytest.raises(ValueError):
        qwen1000_to_pixels((0, 0, 1001, 1000), 200, 80)


def test_ap_matches_hand_computation():
    assert ap101([1], 1) == 1
    assert ap101([0, 1], 1) == 0.5
    assert ap101([1, 0], 1) == 1
    assert ap101([1], 2) == pytest.approx(51 / 101)
    assert ap101([], 1) == 0


def test_classification_failure_denominator(protocol):
    rows = [sample(label="飞机"), sample("b", label="车辆"), sample("c", label="飞机")]
    preds = [Prediction(id="a", label="飞机"), Prediction(id="b", label="飞机")]
    r = evaluate(rows, preds, protocol)["classification"]
    assert r["accuracy"] == pytest.approx(1 / 3)
    assert r["balanced_accuracy"] == 0.25
    assert r["confusion_matrix"] == [[1, 0, 1], [1, 0, 0]]
    assert r["invalid_or_abstained"] == 1
    assert r["per_class"]["飞机"]["f1"] == 0.5


def test_perfect_detection_and_negative_false_positive(protocol):
    rows = [sample(objects=[box()]), sample("empty", objects=[])]
    pred = [Prediction(id="a", objects=[box(0.8)]), Prediction(id="empty", objects=[box(0.9)])]
    r = evaluate(rows, pred, protocol)["detection"]
    assert r["AP50"] == 0.5 and r["AP75"] == 0.5
    assert r["mAP50_95"] == 0.5
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 0
    assert r["false_positives_per_image"] == 0.5


def test_duplicate_detection_is_fp(protocol):
    r = evaluate(
        [sample(objects=[box()])], [Prediction(id="a", objects=[box(0.9), box(0.8)])], protocol
    )["detection"]
    assert r["AP50"] == 1
    assert r["precision"] == 0.5 and r["recall"] == 1


def test_missed_detection_and_absent_class(protocol):
    rows = [sample(objects=[box(), box(xy=(30.0, 30.0, 40.0, 40.0))])]
    r = evaluate(rows, [Prediction(id="a", objects=[box(), box(0.8, label="车辆")])], protocol)[
        "detection"
    ]
    assert r["AP50"] == pytest.approx(51 / 101)
    assert r["fp"] == 1 and r["fn"] == 1
    assert r["per_class"]["车辆"]["AP50"] is None


def test_no_annotation_vs_background(protocol):
    assert evaluate([sample(label="飞机")], [], protocol)["detection"]["status"] == "NOT_ANNOTATED"
    r = evaluate([sample(objects=[])], [Prediction(id="a", objects=[box()])], protocol)
    assert r["detection"]["fp"] == 1
    assert r["detection"]["AP50"] is None


def test_score_threshold_does_not_change_ap(protocol):
    p = protocol.model_copy(update={"score_threshold": 0.8})
    r = evaluate([sample(objects=[box()])], [Prediction(id="a", objects=[box(0.5)])], p)
    assert r["detection"]["AP50"] == 1
    assert r["detection"]["fn"] == 1


def test_unscored_boxes_do_not_invent_ap(protocol):
    rows, preds = [sample(objects=[box()])], [Prediction(id="a", objects=[box(None)])]
    r = evaluate(rows, preds, protocol)["detection"]
    assert r["recall"] == 1 and r["AP50"] is None
    protocol.score_threshold = 0.5
    assert evaluate(rows, preds, protocol)["detection"]["status"] == "MISSING_SCORES"


def test_iou_sweep_and_wrong_class(protocol):
    r = evaluate(
        [sample(objects=[box()])],
        [Prediction(id="a", objects=[box(xy=(0.0, 0.0, 30.0, 20.0))])],
        protocol,
    )["detection"]
    assert r["AP50"] == 1 and r["AP75"] == 0
    assert r["mAP50_95"] == 0.4
    r = evaluate(
        [sample(objects=[box()])], [Prediction(id="a", objects=[box(label="车辆")])], protocol
    )["detection"]
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 1


def test_description_chinese_and_multireference(protocol):
    assert text_pair("强散射，阴影明显。", "强散射阴影明显")["rouge_l_char_f1"] == 1
    assert text_pair("甲乙丙", "丙乙甲")["char_f1"] == 1
    assert text_pair("甲乙丙", "丙乙甲")["rouge_l_char_f1"] == pytest.approx(1 / 3)
    r = evaluate(
        [sample(references=["阴影明显", "强散射"])],
        [Prediction(id="a", description="强散射")],
        protocol,
    )
    assert r["description"]["rouge_l_char_f1"] == 1
    assert evaluate([sample(references=["强散射"])], [], protocol)["description"]["char_f1"] == 0


def test_attribute_contradiction_and_unknown(protocol):
    rows = [
        sample(attributes={"阴影": "明显"}),
        sample("b", attributes={"阴影": None}),
        sample("c", attributes={"阴影": "不明显"}),
    ]
    preds = [
        Prediction(id="a", attributes={"阴影": "不明显"}),
        Prediction(id="b", attributes={"阴影": "明显"}),
    ]
    r = evaluate(rows, preds, protocol)["attributes"]
    assert r["contradictions"] == 1 and r["omitted_or_abstained"] == 1
    assert r["claims_on_unknown"] == 1 and r["accuracy_known"] == 0
    assert r["contradiction_rate_answered"] == 1


def test_blind_prompt_and_exact_label_parser(protocol):
    row = sample(label="飞机")
    assert "image.png" not in prompt("classification", protocol)
    assert parse_output("飞机", row, "classification", protocol).label == "飞机"
    with pytest.raises((ValueError, TypeError)):
        parse_output("这是飞机，因为真值是飞机", row, "classification", protocol)
    assert parse_output("null", row, "classification", protocol).label is None


def test_detection_parser_single_conversion(protocol):
    raw = json.dumps(
        {"objects": [{"label": "飞机", "bbox_1000": [0, 0, 200, 200], "raw_score": 0.9}]}
    )
    p = parse_output(raw, sample(objects=[]), "detection", protocol)
    assert p.objects[0].bbox_xyxy == (0, 0, 20, 20)
    with pytest.raises(ValueError):
        parse_output(raw.replace("200", "1001"), sample(objects=[]), "detection", protocol)
    with pytest.raises(ValueError):
        parse_output('{"objects":[],"label":"车辆"}', sample(objects=[]), "detection", protocol)


def test_failed_prediction_does_not_get_credit(protocol):
    rows = [sample(label="飞机", objects=[box()], references=["阴影明显"])]
    preds = [
        Prediction(
            id="a", status="parse_error", label="飞机", objects=[box()], description="阴影明显"
        )
    ]
    r = evaluate(rows, preds, protocol)
    assert r["status"] == "INCOMPLETE" and r["failure_rate"] == 1
    assert r["classification"]["accuracy"] == 0 and r["detection"]["fn"] == 1
    assert r["description"]["char_f1"] == 0


def test_duplicate_unknown_and_invalid_boxes_rejected(protocol):
    rows = [sample(label="飞机")]
    with pytest.raises(ValueError):
        evaluate(rows, [Prediction(id="a"), Prediction(id="a")], protocol)
    with pytest.raises(ValueError):
        evaluate(rows, [Prediction(id="x")], protocol)
    with pytest.raises(ValueError):
        evaluate(rows, [Prediction(id="a", objects=[box(xy=(0.0, 0.0, 101.0, 10.0))])], protocol)
    with pytest.raises(ValueError):
        Detection(label="飞机", bbox_xyxy=(0.0, 0.0, 10.0, 10.0), raw_score=float("nan"))


def test_manifest_fixture_guard_and_split(protocol, tmp_path):
    p = tmp_path / "samples.jsonl"
    p.write_text(json_line(sample(label="飞机")))
    with pytest.raises(ValueError):
        read_samples(p, protocol)
    assert len(read_samples(p, protocol, True)) == 1
    p.write_text(p.read_text() + p.read_text())
    with pytest.raises(ValueError):
        read_samples(p, protocol, True)


def test_prepare_csv_leakage_and_test_only(tmp_path):
    Image.new("L", (100, 100), 0).save(tmp_path / "image.png")
    path = tmp_path / "labels.csv"
    fields = ["image", "label", "source_group_id", "split", "annotation_status"]
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, ["image.png", "飞机", "g", "test", "verified"])))
    output = tmp_path / "eval.jsonl"
    assert prepare(path, tmp_path, output) == 1
    with path.open("a") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow(dict(zip(fields, ["image.png", "飞机", "g2", "train", "verified"])))
    with pytest.raises(ValueError, match="leaks"):
        prepare(path, tmp_path, tmp_path / "other.jsonl")


def test_image_size_and_training_leakage(tmp_path):
    Image.new("L", (100, 100)).save(tmp_path / "image.png")
    row = sample(label="飞机")
    assert check_inputs([row], tmp_path / "ref.jsonl", None, None)["a"].is_file()
    excluded = tmp_path / "train.json"
    excluded.write_text(
        json.dumps(
            {"records": [{"split": "train", "source_group_id": "a", "image_sha256": "unknown"}]}
        )
    )
    with pytest.raises(ValueError, match="overlaps"):
        check_inputs([row], tmp_path / "ref.jsonl", None, excluded)
    row.width = 101
    with pytest.raises(ValueError, match="size"):
        check_inputs([row], tmp_path / "ref.jsonl", None, None)


def test_score_cli_without_model_or_gpu(tmp_path, protocol):
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(json_line(sample(label="飞机")))
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json_line(Prediction(id="a", label="飞机")))
    proto = tmp_path / "protocol.json"
    proto.write_text(protocol.model_dump_json())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sar_vlm.evaluation.cli",
            "score",
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--protocol",
            str(proto),
            "--output",
            str(tmp_path / "result"),
            "--allow-fixtures",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "result/metrics.json").read_text())
    assert report["classification"]["accuracy"] == 1
    assert report["scientific_result"] is False
    assert report["provenance"]["inference_executed"] is False
