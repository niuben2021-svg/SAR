"""Check installed SftArguments via its official parser without training side effects."""

import argparse
import importlib.metadata
from pathlib import Path

from sar_vlm.runtime import write_json
from sar_vlm.training.config import load_training_config


def main() -> int:
    """Parse flags/types/choices into Namespace, never instantiate SftArguments.

    SftArguments.__post_init__ may initialize distributed execution, data and models.
    argparse.Namespace validation deliberately does not invoke those operations.
    """
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--report", type=Path, required=True)
    args = cli.parse_args()
    from swift.arguments import SftArguments
    from transformers import HfArgumentParser

    cfg = load_training_config(args.config)
    tokens = []
    for key, value in cfg.model_dump().items():
        if key not in {"nproc_per_node", "image_max_token_num"}:
            tokens.extend(
                [f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)]
            )
    parser = HfArgumentParser(SftArguments)
    parser.parse_args(tokens)
    write_json(
        args.report,
        {
            "status": "PASS",
            "ms_swift_version": importlib.metadata.version("ms-swift"),
            "validator": "HfArgumentParser(SftArguments).parse_args",
            "checked_tokens": tokens,
            "training_executed": False,
            "limitations": [
                "Namespace only; no SftArguments post-init or dataset/model loading.",
                "Does not validate GPU/DDP execution or SFT template integration.",
            ],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
