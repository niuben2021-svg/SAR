"""Coordinate conversions, exactly once at the model-output boundary."""

from sar_vlm.data.schema import Box, validate_box


def qwen1000_to_pixels(box: Box, width: int, height: int) -> Box:
    """Convert Qwen relative-1000 XYXY to original half-open pixel XYXY; never clip."""
    validate_box(box, 1000, 1000)
    result = (
        box[0] * width / 1000,
        box[1] * height / 1000,
        box[2] * width / 1000,
        box[3] * height / 1000,
    )
    return validate_box(result, width, height)
