"""Convert the training CSV contract to a separate held-out evaluation manifest."""

import csv
import json
from pathlib import Path

from sar_vlm.data.reader import SARReader
from sar_vlm.evaluation.schema import Sample, json_line


def prepare(csv_path: Path, image_root: Path, output: Path, split: str = "test") -> int:
    """Preserve explicit labels; check train/test group and identical-image leakage."""
    if output.exists():
        raise ValueError("output exists; choose a new manifest")
    root = image_root.resolve(strict=True)
    groups, images, rows = {}, {}, []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for n, row in enumerate(csv.DictReader(f), 2):
            if any(
                not row.get(k, "").strip()
                for k in ("image", "source_group_id", "split", "annotation_status")
            ):
                raise ValueError(f"CSV line {n}: missing required columns")
            if (
                row["split"] not in ("train", "val", "test")
                or row["annotation_status"] != "verified"
            ):
                raise ValueError("require verified labels and train/val/test split")
            path = (root / row["image"]).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("image escapes image root")
            record = SARReader().read(path)
            if record.dtype != "uint8":
                raise ValueError("explicit uint8 display conversion required")
            for mapping, key in ((groups, row["source_group_id"]), (images, record.sha256)):
                if key in mapping and mapping[key] != row["split"]:
                    raise ValueError("source group or identical image leaks across splits")
                mapping[key] = row["split"]
            if row["split"] != split:
                continue
            # physical_observation was a training INPUT, never copy it as reference truth.
            payload = {
                "id": row.get("id") or f"csv-{n}",
                "image": str(path),
                "width": record.width,
                "height": record.height,
                "source_group_id": row["source_group_id"],
                "split": split,
                "annotation_status": "verified",
                "image_sha256": record.sha256,
                "label": row.get("label", "").strip() or None,
                "objects": json.loads(row["objects_json"]) if row.get("objects_json") else None,
                "references": json.loads(row["references_json"])
                if row.get("references_json")
                else None,
                "attributes": json.loads(row["attributes_json"])
                if row.get("attributes_json")
                else None,
            }
            rows.append(Sample.model_validate_json(json.dumps(payload)))
    if not rows or len({r.id for r in rows}) != len(rows):
        raise ValueError("no selected rows or duplicate IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json_line(row) for row in rows))
    return len(rows)
