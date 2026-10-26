"""Read image pixels without display normalization or in-place file updates."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from sar_vlm.data.schema import Metadata


def sha256_file(path: Path) -> str:
    """Hash file bytes in bounded-memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ImageRecord:
    """Read-only HxW or HxWxC pixel array in the file's decoded native dtype/range."""

    path: Path
    pixels: np.ndarray
    width: int
    height: int
    channels: int
    mode: str
    dtype: str
    intensity_range: tuple[float, float]
    sha256: str
    metadata: Metadata
    metadata_valid: dict[str, bool]
    display_transform: str = "none"


class SARReader:
    """MVP PIL-readable amplitude/intensity image reader; no SLC interpretation."""

    def read(self, path: Path, metadata: Metadata | None = None) -> ImageRecord:
        """Return read-only pixels; preserve 16-bit intensity and source file bytes."""
        path = path.resolve(strict=True)
        try:
            with Image.open(path) as image:
                image.load()
                pixels = np.array(image)
                mode = image.mode
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"cannot decode image: {path}") from exc
        if mode not in {"L", "I;16", "I;16L", "I;16B", "I", "F", "RGB", "RGBA"}:
            raise ValueError(f"unsupported pixel mode {mode}; convert a separate display copy")
        if not np.isfinite(pixels).all():
            raise ValueError("image contains nonfinite intensities")
        pixels.setflags(write=False)
        meta = metadata or Metadata()
        return ImageRecord(
            path=path,
            pixels=pixels,
            width=pixels.shape[1],
            height=pixels.shape[0],
            channels=1 if pixels.ndim == 2 else pixels.shape[2],
            mode=mode,
            dtype=str(pixels.dtype),
            intensity_range=(float(pixels.min()), float(pixels.max())),
            sha256=sha256_file(path),
            metadata=meta,
            metadata_valid={
                k: v is not None for k, v in meta.model_dump().items() if k != "provenance"
            },
        )
