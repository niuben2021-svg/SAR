"""Evaluation-only contracts: absent labels are distinct from verified negatives."""

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from sar_vlm.data.schema import Box, Record, validate_box


class Detection(Record):
    """Horizontal box in original pixels; score is an uncalibrated ranking value."""

    label: str = Field(min_length=1)
    bbox_xyxy: Box
    raw_score: float | None = Field(default=None, ge=0, le=1)


class Sample(Record):
    """A held-out sample; null objects means unannotated, [] means verified background."""

    id: str = Field(min_length=1)
    image: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source_group_id: str = Field(min_length=1)
    split: Literal["test", "val", "fixture"]
    annotation_status: Literal["verified", "synthetic_fixture"]
    image_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    label: str | None = None
    objects: list[Detection] | None = None
    references: list[str] | None = None
    attributes: dict[str, str | None] | None = None

    @model_validator(mode="after")
    def validate_annotations(self) -> "Sample":
        """Reject invalid boxes, ambiguous empty references and mixed provenance."""
        if (self.split == "fixture") != (self.annotation_status == "synthetic_fixture"):
            raise ValueError("fixture split and provenance must occur together")
        for obj in self.objects or []:
            validate_box(obj.bbox_xyxy, self.width, self.height)
        if self.references is not None and (
            not self.references or any(not x.strip() for x in self.references)
        ):
            raise ValueError("references must be null or contain nonempty reference descriptions")
        if self.label is not None and not self.label.strip():
            raise ValueError("label must be nonempty or null")
        if all(x is None for x in (self.label, self.objects, self.references, self.attributes)):
            raise ValueError("at least one supervised target is required")
        return self


class Prediction(Record):
    """Normalized external or generated prediction; failed outputs are retained."""

    id: str
    status: Literal["ok", "parse_error", "inference_error", "missing"] = "ok"
    label: str | None = None
    objects: list[Detection] = Field(default_factory=list)
    description: str = ""
    attributes: dict[str, str | None] = Field(default_factory=dict)
    raw_text: str = ""
    error: str | None = None
    trace: dict = Field(default_factory=dict)


class Protocol(Record):
    """Freeze vocabulary/operating threshold before examining test outcomes."""

    classes: list[str]
    attribute_values: dict[str, list[str]] = Field(default_factory=dict)
    score_threshold: float = Field(default=0.0, ge=0, le=1)
    detection_iou: float = Field(default=0.5, gt=0, le=1)
    max_detections: int = Field(default=100, gt=0, le=1000)

    @model_validator(mode="after")
    def vocabulary(self) -> "Protocol":
        """Require unique meaningful labels and closed attribute vocabularies."""
        if not self.classes or any(not x.strip() for x in self.classes):
            raise ValueError("classes must be nonempty")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("duplicate class")
        if any(
            not k.strip() or not v or len(set(v)) != len(v) or any(not x.strip() for x in v)
            for k, v in self.attribute_values.items()
        ):
            raise ValueError("invalid attribute vocabulary")
        return self


def read_samples(path: Path, protocol: Protocol, fixtures: bool = False) -> list[Sample]:
    """Validate unique IDs, vocabulary and one provenance/split stratum per run."""
    rows = [Sample.model_validate_json(s) for s in path.read_text().splitlines() if s.strip()]
    if not rows or len({r.id for r in rows}) != len(rows):
        raise ValueError("manifest must be nonempty with unique IDs")
    if len({r.split for r in rows}) != 1:
        raise ValueError("do not mix test/val/fixture strata")
    if any(r.split == "fixture" for r in rows) and not fixtures:
        raise ValueError("synthetic fixtures require --allow-fixtures; never a scientific result")
    for row in rows:
        labels = ([row.label] if row.label is not None else []) + [
            o.label for o in row.objects or []
        ]
        if any(x not in protocol.classes for x in labels):
            raise ValueError(f"{row.id}: target label not in protocol.classes")
        for k, v in (row.attributes or {}).items():
            if k not in protocol.attribute_values or (
                v is not None and v not in protocol.attribute_values[k]
            ):
                raise ValueError(f"{row.id}: invalid reference attribute {k}")
    return rows


def read_predictions(path: Path) -> list[Prediction]:
    """Read normalized predictions without guessing IDs, labels or coordinates."""
    return [Prediction.model_validate_json(s) for s in path.read_text().splitlines() if s.strip()]


def json_line(value: Record) -> str:
    """Encode a validated record as one UTF-8 JSONL line."""
    return json.dumps(value.model_dump(mode="json"), ensure_ascii=False) + "\n"
