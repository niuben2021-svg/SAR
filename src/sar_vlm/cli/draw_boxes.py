"""Render externally supplied pixel boxes to a separate PNG; no inference."""

import argparse
import json
from pathlib import Path

from sar_vlm.inference.result_writer import draw_boxes


def main() -> None:
    """Draw explicit engineering/annotation boxes from a JSON array."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True, help="JSON array of pixel XYXY boxes")
    parser.add_argument("--width", type=int, default=3)
    args = parser.parse_args()
    boxes = json.loads(args.boxes.read_text(encoding="utf-8"))
    draw_boxes(args.image, args.output, boxes, args.width)


if __name__ == "__main__":
    main()
