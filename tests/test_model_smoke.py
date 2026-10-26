"""CPU interface checks are distinct from explicitly opted-in real GPU execution."""

import os
from pathlib import Path

import pytest

from sar_vlm.config import load_config
from sar_vlm.models.qwen_adapter import QwenAdapter

ROOT = Path(__file__).resolve().parents[1]


def test_adapter_import_has_no_weight_allocation():
    adapter = QwenAdapter(load_config(ROOT / "configs/mvp.yaml"))
    assert adapter.model is None and adapter.processor is None


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("SAR_RUN_GPU_SMOKE") != "1",
    reason="opt-in real Qwen3VL weights/GPU smoke: SAR_RUN_GPU_SMOKE=1 and SAR_IMAGE",
)
def test_real_qwen_forward():
    adapter = QwenAdapter(load_config(ROOT / "configs/mvp.yaml"))
    adapter.load()
    result = adapter.generate(Path(os.environ["SAR_IMAGE"]), "请描述图像中可观测的内容")
    assert result["inference_executed"]
    assert result["raw_text"].strip()
    assert "pixel_values" in result["input_shapes"]
    assert result["gpu_peak_memory_bytes"] > 0
