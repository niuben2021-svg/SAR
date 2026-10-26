"""User-started LoRA training, following a completed single-GPU smoke."""

from sar_vlm.training.offline_smoke import main

if __name__ == "__main__":
    raise SystemExit(main(full=True))
