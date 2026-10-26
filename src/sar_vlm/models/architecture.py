"""Supported dense Qwen3-VL sizes; configuration checks do not validate weight contents."""

SUPPORTED_MODELS = {"4B": (2560, 36), "8B": (4096, 32)}
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def model_size(config: dict) -> str:
    """Identify supported text dimensions and reject MoE/quantized model variants."""
    text = config.get("text_config", {})
    dimensions = (text.get("hidden_size"), text.get("num_hidden_layers"))
    if config.get("model_type") == "qwen3_vl" and not config.get("quantization_config"):
        for size, expected in SUPPORTED_MODELS.items():
            if dimensions == expected:
                return size
    raise ValueError("expected unquantized dense Qwen3-VL-4B or Qwen3-VL-8B architecture")
