"""Validated configuration; paths resolve relative to the YAML file."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MVPConfig(BaseModel):
    """MVP runtime settings; no model is loaded while parsing configuration."""

    model_config = ConfigDict(extra="forbid")
    model_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    revision: str = "main"
    local_files_only: bool = True
    device: str = "cuda:0"
    seed: int = Field(default=42, ge=0)
    image_max_tokens: int = Field(default=1024, ge=4)
    max_new_tokens: int = Field(default=128, gt=0, le=1024)
    input_image: Path | None = None
    manifest: Path | None = None
    output_dir: Path
    report_dir: Path

    @field_validator("device")
    @classmethod
    def device_valid(cls, value: str) -> str:
        """Require a supported single-device target."""
        if value != "cpu" and not (value.startswith("cuda:") and value[5:].isdigit()):
            raise ValueError("device must be cpu or cuda:<index>")
        return value


def load_config(path: Path) -> MVPConfig:
    """Load YAML, resolve local paths, and reject missing input files."""
    path = path.resolve(strict=True)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    cfg = MVPConfig.model_validate(raw)
    for key in ("input_image", "manifest", "output_dir", "report_dir"):
        value = getattr(cfg, key)
        if value is not None:
            value = (path.parent / value).resolve()
            if key in ("input_image", "manifest") and not value.is_file():
                raise FileNotFoundError(f"{key}: {value}")
            setattr(cfg, key, value)
    if cfg.model_path not in {"Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"}:
        cfg.model_path = str((path.parent / cfg.model_path).resolve(strict=True))
    return cfg
