"""Separate blind prompts, raw generation, validated predictions and supervised scoring."""

import csv
import json
from pathlib import Path

from sar_vlm.data.conversion import qwen1000_to_pixels
from sar_vlm.data.schema import validate_box
from sar_vlm.evaluation import metrics
from sar_vlm.evaluation.schema import Prediction, Protocol, Sample


def prompt(task: str, protocol: Protocol) -> str:
    """Build a blind shared prompt from vocabulary only; never include sample annotations."""
    common = (
        "你正在分析SAR显示图像。仅报告图像可支持的观测，不要从幅度图推断相位或极化。"
        "类别只能使用以下词表：" + json.dumps(protocol.classes, ensure_ascii=False) + "。"
    )
    if task == "classification":
        return common + "请识别主要目标类别，仅输出一个类别名称；无法判断则输出 null。"
    fields = {}
    if task in ("detection", "all"):
        fields["objects"] = [
            {"label": "词表中的类别", "bbox_1000": [0, 0, 1000, 1000], "raw_score": 0.5}
        ]
        common += (
            "请检测图中所有词表内目标。框为相对原图的0到1000归一化XYXY，"
            "raw_score为你对该检测的0到1排序估计，不是校准概率。没有目标则objects=[]。"
        )
    if task in ("description", "all"):
        fields["description"] = "可观察的形态和散射强度分布描述"
        fields["attributes"] = {k: None for k in protocol.attribute_values}
        common += (
            "属性允许值："
            + json.dumps(protocol.attribute_values, ensure_ascii=False)
            + "。无法确定的属性填null，不能编造无法观测的物理量。"
        )
    if task == "all":
        fields["label"] = "主要目标类别，无法判断则null"
    return (
        common
        + "只输出JSON对象，不要Markdown或解释，结构为："
        + json.dumps(fields, ensure_ascii=False)
    )


def validate_prediction(pred: Prediction, sample: Sample, protocol: Protocol) -> None:
    """Reject invented classes/attributes and out-of-bounds boxes rather than clipping."""
    if pred.status != "ok":
        return
    if pred.label is not None and pred.label not in protocol.classes:
        raise ValueError("predicted label outside vocabulary")
    for obj in pred.objects:
        if obj.label not in protocol.classes:
            raise ValueError("detection class outside vocabulary")
        validate_box(obj.bbox_xyxy, sample.width, sample.height)
    for key, value in pred.attributes.items():
        if key not in protocol.attribute_values or (
            value is not None and value not in protocol.attribute_values[key]
        ):
            raise ValueError("predicted attribute outside vocabulary")


def parse_output(raw: str, sample: Sample, task: str, protocol: Protocol) -> Prediction:
    """Parse complete JSON or exact classification label; no GT-dependent recovery."""
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    if task == "classification" and (text in protocol.classes or text == "null"):
        value = {"label": None if text == "null" else text}
    else:
        value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    required = {
        "classification": {"label"},
        "detection": {"objects"},
        "description": {"description", "attributes"},
        "all": {"label", "objects", "description", "attributes"},
    }[task]
    if set(value) != required:
        raise ValueError(f"output must have exactly these keys: {sorted(required)}")
    if "objects" in value:
        converted = []
        for obj in value["objects"]:
            if set(obj) not in ({"label", "bbox_1000", "raw_score"}, {"label", "bbox_1000"}):
                raise ValueError("invalid detection fields")
            converted.append(
                {
                    "label": obj["label"],
                    "raw_score": obj.get("raw_score"),
                    "bbox_xyxy": qwen1000_to_pixels(
                        tuple(obj["bbox_1000"]), sample.width, sample.height
                    ),
                }
            )
        value["objects"] = converted
    pred = Prediction.model_validate_json(json.dumps({"id": sample.id, "raw_text": raw, **value}))
    validate_prediction(pred, sample, protocol)
    return pred


def evaluate(
    samples: list[Sample], rows: list[Prediction], protocol: Protocol, task: str = "all"
) -> dict:
    """Score every labeled item, retaining failures and negatives in the denominator."""
    predictions = {p.id: p for p in rows}
    if len(predictions) != len(rows):
        raise ValueError("duplicate prediction IDs")
    if set(predictions) - {s.id for s in samples}:
        raise ValueError("prediction IDs absent from reference manifest")
    for sample in samples:
        if sample.id not in predictions:
            predictions[sample.id] = Prediction(id=sample.id, status="missing")
        validate_prediction(predictions[sample.id], sample, protocol)
    failures = [
        {"id": p.id, "status": p.status, "error": p.error}
        for p in predictions.values()
        if p.status != "ok"
    ]
    result = {
        "status": "INCOMPLETE" if failures else "OK",
        "samples": len(samples),
        "failed_samples": len(failures),
        "failure_rate": len(failures) / len(samples),
        "failures": failures,
        "task": task,
        "protocol": protocol.model_dump(),
        "scientific_result": samples[0].split != "fixture",
        "split": samples[0].split,
        "ranking_note": "Generated detection scores are self-reported, uncalibrated "
        "ranking estimates. AP is an exploratory grounding baseline; "
        "no calibrated confidence or trained detector is implied.",
    }
    selected = {
        "classification": metrics.classification,
        "detection": metrics.detection,
        "description": metrics.descriptions,
        "attributes": metrics.attributes,
    }
    for name, function in selected.items():
        enabled = task == "all" or name == task or (name == "attributes" and task == "description")
        if not enabled:
            result[name] = {"status": "NOT_REQUESTED"}
        elif name == "classification":
            result[name] = function(samples, predictions, protocol.classes)
        elif name == "detection":
            result[name] = function(samples, predictions, protocol)
        else:
            result[name] = function(samples, predictions)
    return result


def write_reports(output: Path, report: dict) -> None:
    """Write machine-readable metrics plus a readable summary without nonfinite JSON."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    text = "# SAR 批量评估结果\n\n"
    text += "这是工程合成测试，不代表SAR精度。\n\n" if not report["scientific_result"] else ""
    text += (
        "失败样本计入分母，未静默丢弃。状态：" + report["status"] + "。\n\n"
        "检测为生成式水平框基线；AP使用未校准排序分数。文本重合不证明物理正确。\n\n"
    )
    for key in ("classification", "detection", "description", "attributes"):
        text += (
            f"## {key}\n\n```json\n"
            + json.dumps(report[key], ensure_ascii=False, indent=2)
            + "\n```\n\n"
        )
    (output / "report.md").write_text(text)
    classification = report["classification"]
    if classification.get("status") == "OK":
        with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["true/pred", *classification["column_labels"]])
            for label, values in zip(
                classification["row_labels"], classification["confusion_matrix"]
            ):
                writer.writerow([label, *values])
