"""Canonical v0.1 records in original-image pixel XYXY, right/bottom exclusive."""

import json
import math
from numbers import Real
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

Status = Literal[
    "verified", "weak", "candidate", "synthetic_fixture", "illustrative_only", "pending"
]
Box = tuple[float, float, float, float]


def validate_box(box: Box, width: int, height: int) -> Box:
    """Reject nonfinite, empty or out-of-image half-open pixel XYXY; never clip."""
    if len(box) != 4 or not all(
        isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(v) for v in box
    ):
        raise ValueError("bbox_xyxy must contain four finite pixel coordinates")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"bbox_xyxy outside {width}x{height} or has nonpositive area: {box}")
    return box


class Record(BaseModel):
    """Strict field names and types avoid silent annotation coercion."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Metadata(Record):
    """Unknown physical metadata remain null; no SLC evidence is inferred."""

    sensor: str | None = None
    acquisition_time: str | None = None
    center_frequency_hz: float | None = Field(default=None, gt=0)
    incidence_angle_deg: float | None = Field(default=None, ge=0, le=90)
    look_azimuth_deg: float | None = Field(default=None, ge=0, lt=360)
    range_resolution_m: float | None = Field(default=None, gt=0)
    azimuth_resolution_m: float | None = Field(default=None, gt=0)
    pixel_spacing_m: float | None = Field(default=None, gt=0)
    polarization: str | None = None
    product_type: str | None = None
    provenance: str = "unknown"


class Query(Record):
    """Natural-language query with explicit target class vocabulary."""

    text: str = Field(min_length=1)
    target_classes: list[str]


class Evidence(Record):
    """Spatially grounded evidence with explicit verification and observability."""

    type: str
    name: str
    bbox_xyxy: Box
    observability: Literal["observable", "unobservable", "unknown"] = "unknown"
    verification: Status = "pending"


class ObjectRecord(Record):
    """A labeled object; annotation provenance is always required."""

    object_id: str = Field(min_length=1)
    bbox_xyxy: Box
    coarse_class: str = Field(min_length=1)
    fine_class: str | None = None
    annotation_status: Status
    evidence: list[Evidence] = Field(default_factory=list)


class SceneRegion(Record):
    """An optional scene region in original pixel coordinates."""

    region_id: str
    label: str
    bbox_xyxy: Box
    verification: Status


class CanonicalSample(Record):
    """Minimal data contract; images resolve relative to their JSONL manifest."""

    schema_version: Literal["0.1"]
    scene_id: str = Field(min_length=1)
    source_image: str = Field(min_length=1)
    width: PositiveInt
    height: PositiveInt
    image_type: Literal["SAR_AMPLITUDE", "SAR_INTENSITY", "SYNTHETIC_FIXTURE"]
    source_group_id: str = Field(min_length=1)
    split: Literal["train", "val", "test", "fixture"]
    query: Query
    metadata: Metadata = Field(default_factory=Metadata)
    scene_regions: list[SceneRegion] = Field(default_factory=list)
    objects: list[ObjectRecord]
    # Historical registration/time semantics are intentionally deferred to KNOW-01.
    historical_priors: list[dict] = Field(default_factory=list)
    label_provenance: Status
    image_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    license: str | None = None

    @model_validator(mode="after")
    def check_coordinates(self) -> "CanonicalSample":
        """Validate every spatial annotation and reject duplicate object IDs."""
        ids = [obj.object_id for obj in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("object_id must be unique within a scene")
        for item in [*self.objects, *self.scene_regions]:
            validate_box(item.bbox_xyxy, self.width, self.height)
        for obj in self.objects:
            for evidence in obj.evidence:
                validate_box(evidence.bbox_xyxy, self.width, self.height)
        if self.historical_priors:
            raise ValueError("historical_priors require KNOW-01 validation; disabled in MVP")
        if self.image_type == "SYNTHETIC_FIXTURE" and (
            self.split != "fixture"
            or self.label_provenance != "synthetic_fixture"
            or any(obj.annotation_status != "synthetic_fixture" for obj in self.objects)
        ):
            raise ValueError("engineering synthetic fixtures must keep fixture split/provenance")
        return self


def read_manifest(path: Path) -> list[CanonicalSample]:
    """Read nonempty JSONL; include file and line in validation errors."""
    samples = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            samples.append(CanonicalSample.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    if not samples:
        raise ValueError(f"manifest is empty: {path}")
    return samples


def write_manifest(path: Path, sample: CanonicalSample) -> None:
    """Write one validated canonical engineering sample, without modifying images."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n", encoding="utf-8"
    )
