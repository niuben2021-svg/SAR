"""Explicit real-weight forward entrypoint; failures remain nonzero and recorded."""

import argparse
import logging
import traceback
from pathlib import Path

from sar_vlm.config import load_config
from sar_vlm.models.qwen_adapter import ModelUnavailable, QwenAdapter
from sar_vlm.runtime import command, config_hash, timestamp, write_json


def main() -> int:
    """Run pretrained image+text generation or persist an honest BLOCKED/FAILED trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    base = {
        "timestamp": timestamp(),
        "config_hash": config_hash(args.config),
        "git": command(["git", "rev-parse", "HEAD"]),
        "weights": cfg.model_path,
        "requested_revision": cfg.revision,
        "device": cfg.device,
        "inference_executed": False,
        "gpu_peak_memory_bytes": None,
    }
    try:
        image = args.image or cfg.input_image
        if image is None or not image.is_file():
            raise FileNotFoundError("provide --image or a valid input_image")
        adapter = QwenAdapter(cfg)
        adapter.load()
        result = adapter.generate(image, args.query)
        write_json(cfg.output_dir / "forward_output.json", {**base, **result, "status": "PASS"})
        logging.info("Real Qwen forward completed")
        return 0
    except (ModelUnavailable, ImportError, OSError) as exc:
        status, code = "BLOCKED", 2
        error = exc
    except Exception as exc:
        status, code = "FAILED", 1
        error = exc
    # Keep failure separate from forward_output.json; never fabricate raw_text/detections.
    write_json(
        cfg.output_dir / "forward_failure.json",
        {
            **base,
            "status": status,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        },
    )
    logging.error("%s: %s", status, error)
    return code


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
