"""Official Qwen3-VL adapter; lazy imports keep CPU data checks model-independent."""

import json
import time
from pathlib import Path

from sar_vlm.config import MVPConfig
from sar_vlm.data.reader import SARReader, sha256_file
from sar_vlm.models.architecture import model_size


class ModelUnavailable(RuntimeError):
    """An actual dependency, device or model prerequisite is unavailable."""


def require_device(device: str) -> None:
    """Check device/BF16 availability before any model weight access."""
    try:
        import torch
    except ImportError as exc:
        raise ModelUnavailable("PyTorch absent; install the compatible model extra") from exc
    if device.startswith("cuda:"):
        index = int(device.split(":")[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ModelUnavailable(
                f"requested {device}, visible CUDA GPUs={torch.cuda.device_count()}"
            )
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise ModelUnavailable(f"{device} does not support the configured BF16 workflow")


class QwenAdapter:
    """Single-device pretrained Qwen3-VL-4B/8B generation, never a detector substitute."""

    def __init__(self, config: MVPConfig, adapter_path: Path | None = None) -> None:
        """Store config without allocating weights or touching the network."""
        self.config = config
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None
        self.load_seconds: float | None = None

    def load(self) -> None:
        """Load official pretrained weights with SDPA and verified library defaults."""
        require_device(self.config.device)
        import torch
        from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

        cfg = self.config
        options = {
            "revision": cfg.revision,
            "local_files_only": cfg.local_files_only,
            "trust_remote_code": False,
        }
        architecture = AutoConfig.from_pretrained(cfg.model_path, **options)
        model_size(architecture.to_dict())
        start = time.perf_counter()
        if cfg.device.startswith("cuda:"):
            torch.cuda.reset_peak_memory_stats(cfg.device)
        # Qwen3-VL patch_size=16, spatial_merge_size=2: each visual token covers 32x32.
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_path,
            min_pixels=4 * 32 * 32,
            max_pixels=cfg.image_max_tokens * 32 * 32,
            **options,
        )
        self.model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                cfg.model_path,
                dtype=torch.bfloat16 if cfg.device.startswith("cuda:") else torch.float32,
                attn_implementation="sdpa",
                **options,
            )
            .to(cfg.device)
            .eval()
        )
        if self.adapter_path is not None:
            from peft import PeftModel

            adapter_path = self.adapter_path.resolve(strict=True)
            adapter_config = json.loads((adapter_path / "adapter_config.json").read_text())
            if adapter_config.get("peft_type") != "LORA":
                raise ValueError("evaluation supports PEFT LoRA adapter checkpoints only")
            self.model = PeftModel.from_pretrained(
                self.model, str(adapter_path), is_trainable=False, local_files_only=True
            ).eval()
        if cfg.device.startswith("cuda:"):
            torch.cuda.synchronize(cfg.device)
        self.load_seconds = time.perf_counter() - start

    def generate(self, image: Path, query: str) -> dict:
        """Run real image+text generation; return raw text and trace, no guessed boxes."""
        import torch
        from PIL import Image

        if self.model is None or self.processor is None:
            raise RuntimeError("load() must succeed before generate()")
        if not query.strip():
            raise ValueError("query must not be blank")
        record = SARReader().read(image)
        if record.dtype != "uint8":
            raise ValueError("explicit display conversion is required for non-8-bit model input")
        torch.manual_seed(self.config.seed)
        if self.config.device.startswith("cuda:"):
            torch.cuda.manual_seed_all(self.config.seed)
        # RGB conversion is a separate in-memory model copy, not an on-disk update.
        visual = Image.fromarray(record.pixels).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": visual}, {"type": "text", "text": query}],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.config.device)
        input_shapes = {
            key: list(value.shape) for key, value in inputs.items() if hasattr(value, "shape")
        }
        if self.config.device.startswith("cuda:"):
            torch.cuda.synchronize(self.config.device)
        start = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.config.max_new_tokens
            )
        if self.config.device.startswith("cuda:"):
            torch.cuda.synchronize(self.config.device)
        elapsed = time.perf_counter() - start
        trimmed = [output[len(prompt) :] for prompt, output in zip(inputs.input_ids, generated)]
        raw = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        provenance_path = Path(self.config.model_path) / "download_source.json"
        manifest_path = Path(self.config.model_path) / "transfer_manifest.json"
        return {
            "raw_text": raw,
            "detections": [],
            "calibrated_confidence": None,
            "inference_executed": True,
            "input_shapes": input_shapes,
            "image": str(record.path),
            "image_size": [record.width, record.height],
            "image_sha256": record.sha256,
            "query": query,
            "device": self.config.device,
            "model_class": type(self.model).__name__,
            "weights": self.config.model_path,
            "requested_revision": self.config.revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "download_provenance": json.loads(provenance_path.read_text())
            if provenance_path.is_file()
            else None,
            "weights_manifest_sha256": sha256_file(manifest_path)
            if manifest_path.is_file()
            else None,
            "load_seconds": self.load_seconds,
            "generation_seconds": elapsed,
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(self.config.device)
            if self.config.device.startswith("cuda:")
            else None,
            "seed": self.config.seed,
            "determinism_guaranteed": False,
        }
