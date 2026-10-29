"""Create clearly marked synthetic metric fixtures; no model inference or SAR claims."""

import argparse
from pathlib import Path

from PIL import Image

from sar_vlm.evaluation.schema import Detection, Prediction, Protocol, Sample, json_line


def main() -> None:
    """Generate two fixtures and intentionally imperfect predictions for a CPU dry run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    Image.new("L", (100, 100), 50).save(args.output / "fixture.png")
    target = Detection(label="飞机", bbox_xyxy=(10.0, 10.0, 30.0, 30.0))
    rows = [
        Sample(
            id="fixture-1",
            image="fixture.png",
            width=100,
            height=100,
            source_group_id="fixture-only",
            split="fixture",
            annotation_status="synthetic_fixture",
            label="飞机",
            objects=[target],
            references=["亮散射点周围存在阴影。"],
            attributes={"阴影": "明显"},
        ),
        Sample(
            id="fixture-2",
            image="fixture.png",
            width=100,
            height=100,
            source_group_id="fixture-only",
            split="fixture",
            annotation_status="synthetic_fixture",
            label="车辆",
            objects=[],
        ),
    ]
    predictions = [
        Prediction(
            id="fixture-1",
            label="飞机",
            objects=[target.model_copy(update={"raw_score": 0.9})],
            description="亮散射点周围存在阴影。",
            attributes={"阴影": "明显"},
        ),
        Prediction(
            id="fixture-2", label="飞机", objects=[target.model_copy(update={"raw_score": 0.95})]
        ),
    ]
    (args.output / "samples.jsonl").write_text("".join(json_line(r) for r in rows))
    (args.output / "predictions.jsonl").write_text("".join(json_line(r) for r in predictions))
    (args.output / "protocol.json").write_text(
        Protocol(
            classes=["飞机", "车辆"], attribute_values={"阴影": ["明显", "不明显"]}
        ).model_dump_json(indent=2)
        + "\n"
    )
    print("ENGINEERING FIXTURE ONLY. Expected classification accuracy=0.5, detection AP50=0.5.")


if __name__ == "__main__":
    main()
