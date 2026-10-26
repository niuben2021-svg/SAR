"""Exercise reader/schema/rendering with explicit synthetic engineering data."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sar_vlm.config import load_config
from sar_vlm.data.reader import SARReader
from sar_vlm.data.schema import CanonicalSample, read_manifest, write_manifest
from sar_vlm.inference.result_writer import draw_boxes
from sar_vlm.runtime import timestamp, write_json


def make_fixture(output: Path) -> Path:
    """Create deterministic grayscale engineering pixels, never SAR ground truth."""
    output.mkdir(parents=True, exist_ok=True)
    image = output / "synthetic_fixture.png"
    pixels = np.arange(64 * 48, dtype=np.uint16).reshape(48, 64).astype(np.uint8)
    if not image.exists():
        Image.fromarray(pixels).save(image)
    elif not np.array_equal(SARReader().read(image).pixels, pixels):
        raise ValueError(f"refusing to overwrite modified fixture {image}")
    sample = CanonicalSample.model_validate_json(
        json.dumps(
            {
                "schema_version": "0.1",
                "scene_id": "engineering_fixture_0001",
                "source_image": image.name,
                "width": 64,
                "height": 48,
                "image_type": "SYNTHETIC_FIXTURE",
                "source_group_id": "engineering_fixture",
                "split": "fixture",
                "query": {"text": "工程接口检查", "target_classes": []},
                "objects": [],
                "label_provenance": "synthetic_fixture",
                "image_sha256": SARReader().read(image).sha256,
                "license": "generated_engineering_fixture",
            }
        )
    )
    manifest = output / "synthetic_fixture.jsonl"
    if manifest.exists() and read_manifest(manifest) != [sample]:
        raise ValueError(f"refusing to overwrite modified fixture manifest {manifest}")
    write_manifest(manifest, sample)
    return manifest


def main() -> int:
    """Read every manifest image and verify dimensions/hash before reporting success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.input_image and not cfg.manifest:
        parser.error("input_image requires a canonical manifest; no labels are inferred")
    manifest = cfg.manifest or make_fixture(cfg.output_dir)
    traces = []
    for sample in read_manifest(manifest):
        path = (manifest.parent / sample.source_image).resolve(strict=True)
        if cfg.input_image and path != cfg.input_image:
            raise ValueError("input_image disagrees with manifest source_image")
        record = SARReader().read(path, sample.metadata)
        if (record.width, record.height) != (sample.width, sample.height):
            raise ValueError(f"image dimensions disagree with manifest: {path}")
        if sample.image_sha256 and sample.image_sha256 != record.sha256:
            raise ValueError(f"image SHA256 disagrees with manifest: {path}")
        traces.append(
            {
                "scene_id": sample.scene_id,
                "path": str(path),
                "shape": list(record.pixels.shape),
                "dtype": record.dtype,
                "intensity_range": record.intensity_range,
                "sha256": record.sha256,
                "metadata_valid": record.metadata_valid,
                "label_provenance": sample.label_provenance,
            }
        )
    boxed = None
    if cfg.manifest is None:
        # Independent engineering box, explicitly unrelated to any model prediction.
        # Use a fresh output name on repeated runs, preserving prior artifacts.
        import uuid

        boxed = cfg.output_dir / "boxed_fixture.png"
        if boxed.exists():
            boxed = cfg.output_dir / f"boxed_fixture_{uuid.uuid4().hex[:8]}.png"
        draw_boxes(cfg.output_dir / "synthetic_fixture.png", boxed, [(8, 8, 32, 30)])
    write_json(
        cfg.output_dir / "data_trace.json",
        {
            "status": "PASS",
            "timestamp": timestamp(),
            "samples": traces,
            "inference_executed": False,
            "manifest": str(manifest),
            "render_output": str(boxed) if boxed else None,
        },
    )
    print(json.dumps(traces, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
