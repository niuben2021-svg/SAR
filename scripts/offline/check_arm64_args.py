"""Validate the single-A100 profile against installed ms-swift argument fields."""

import dataclasses

import yaml
from swift.arguments import SftArguments

from sar_vlm.training.config import TrainingConfig

with open("configs/qwen3vl4b_lora_arm64_a100.yaml") as f:
    config = TrainingConfig.model_validate(yaml.safe_load(f))
fields = {f.name for f in dataclasses.fields(SftArguments)}
assert set(config.model_dump()) - {"nproc_per_node", "image_max_token_num"} <= fields
assert config.nproc_per_node == 1 and config.attn_impl == "sdpa"
print("ARM64 single-A100 training configuration fields: PASS (training not run)")
