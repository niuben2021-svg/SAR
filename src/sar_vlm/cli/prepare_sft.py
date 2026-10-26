"""Convert explicitly reviewed CSV labels to local ms-swift image/text JSONL."""

import argparse
import csv
import json
from pathlib import Path

from sar_vlm.data.reader import SARReader, sha256_file
from sar_vlm.runtime import write_json


def prepare(
    csv_path: Path, image_root: Path, output: Path, reviewer: str, source: str, license_note: str
) -> dict:
    """Validate user-attested labels and group/image separation, without inferring labels."""
    if not all(value.strip() for value in (reviewer, source, license_note)):
        raise ValueError("reviewer, source and license note must be explicit")
    if output.exists():
        raise ValueError("output directory already exists; use a new directory")
    root = image_root.resolve(strict=True)
    grouped = {split: [] for split in ("train", "val", "test")}
    group_splits, image_splits = {}, {}
    records = []
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for number, row in enumerate(csv.DictReader(stream), 2):
            required = ("image", "label", "source_group_id", "split", "annotation_status")
            if any(not row.get(key, "").strip() for key in required):
                raise ValueError(f"CSV line {number}: missing required fields {required}")
            split, group = row["split"], row["source_group_id"]
            if split not in grouped or row["annotation_status"] != "verified":
                raise ValueError(f"CSV line {number}: require train/val/test and verified labels")
            image = (root / row["image"]).resolve(strict=True)
            if not image.is_relative_to(root):
                raise ValueError("image escapes image-root")
            record = SARReader().read(image)
            if record.dtype != "uint8":
                raise ValueError("explicit 8-bit display preparation required before SFT")
            for mapping, key in ((group_splits, group), (image_splits, record.sha256)):
                if key in mapping and mapping[key] != split:
                    raise ValueError("source group or identical image leaks across splits")
                mapping[key] = split
            question = row.get("question", "").strip() or "请识别这幅SAR图像中的目标类别。"
            observation = row.get("physical_observation", "").strip()
            if observation:
                question += "\n已提供的观测信息：" + observation
            grouped[split].append(
                {
                    "messages": [
                        {"role": "user", "content": "<image>" + question},
                        {"role": "assistant", "content": row["label"].strip()},
                    ],
                    "images": [str(image)],
                }
            )
            records.append(
                {
                    "image": str(image),
                    "image_sha256": record.sha256,
                    "source_group_id": group,
                    "split": split,
                    "annotation_status": "verified",
                }
            )
    if not grouped["train"]:
        raise ValueError("CSV has no training samples")
    output.mkdir(parents=True)
    for split, rows in grouped.items():
        if not rows:
            continue
        dataset = output / f"{split}.jsonl"
        dataset.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        write_json(
            output / f"{split}.review.json",
            {
                "annotation_status": "verified",
                "reviewed_by": reviewer,
                "source_dataset": source,
                "license": license_note,
                "sft_sha256": sha256_file(dataset),
                "review_basis": "user-supplied verified CSV; software does not verify label truth",
            },
        )
    result = {
        "counts": {key: len(rows) for key, rows in grouped.items()},
        "csv_sha256": sha256_file(csv_path),
        "records": records,
    }
    write_json(output / "data_manifest.json", result)
    return result


def main() -> None:
    """Prepare classification SFT; input images and labels remain untouched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewed-by", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--license-note", required=True)
    args = parser.parse_args()
    result = prepare(
        args.csv,
        args.image_root,
        args.output,
        args.reviewed_by,
        args.source_dataset,
        args.license_note,
    )
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
