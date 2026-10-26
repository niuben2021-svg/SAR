"""Pixel-preserving red outlines; this utility never creates model detections."""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from sar_vlm.data.reader import SARReader
from sar_vlm.data.schema import Box, validate_box


def draw_boxes(source: Path, destination: Path, boxes: list[Box], width: int = 3) -> Path:
    """Draw red outlines on a new PNG using half-open original pixel XYXY.

    Floats rasterize outwards (floor left/top, ceil right/bottom). For grayscale,
    only an 8-bit RGB display copy is supported: channels equal the original
    grayscale off the outline. Native high-bit-depth inputs are rejected rather
    than silently normalized. Neither the input array nor file is modified.
    """
    if source.resolve() == destination.resolve():
        raise ValueError("destination must differ from source")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    if destination.suffix.lower() != ".png":
        raise ValueError("destination must be PNG to preserve pixels")
    if type(width) is not int or width <= 0:
        raise ValueError("outline width must be a positive integer")
    record = SARReader().read(source)
    if record.dtype != "uint8" or record.mode not in {"L", "RGB", "RGBA"}:
        raise ValueError("pixel-preserving red overlay requires 8-bit L/RGB/RGBA input")
    pixels = record.pixels.copy()
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[:, :, None], 3, axis=2)
    for box in boxes:
        validate_box(box, record.width, record.height)
        x1, y1, x2, y2 = box
        left, top, right, bottom = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
        # Change RGB only; alpha and all pixels outside the outline are preserved.
        pixels[top : min(top + width, bottom), left:right, :3] = (255, 0, 0)
        pixels[max(top, bottom - width) : bottom, left:right, :3] = (255, 0, 0)
        pixels[top:bottom, left : min(left + width, right), :3] = (255, 0, 0)
        pixels[top:bottom, max(left, right - width) : right, :3] = (255, 0, 0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(destination)
    return destination
