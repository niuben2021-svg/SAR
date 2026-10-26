"""Every pixel outside independent half-open outline masks must stay unchanged."""

import numpy as np
import pytest
from PIL import Image

from sar_vlm.data.reader import sha256_file
from sar_vlm.inference.result_writer import draw_boxes


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_only_outline_changes(tmp_path, mode):
    rng = np.random.default_rng(42)
    shape = (30, 40) if mode == "L" else (30, 40, 3 if mode == "RGB" else 4)
    original = rng.integers(0, 256, size=shape, dtype=np.uint8)
    source, output = tmp_path / "source.png", tmp_path / "output.png"
    Image.fromarray(original).save(source)
    digest = sha256_file(source)
    draw_boxes(source, output, [(4, 5, 21, 24), (39, 0, 40, 2)], width=3)
    result = np.array(Image.open(output))
    expected = np.repeat(original[:, :, None], 3, axis=2) if mode == "L" else original
    mask = np.zeros((30, 40), dtype=bool)
    for y in range(30):
        for x in range(40):
            inside = 4 <= x < 21 and 5 <= y < 24
            edge = min(x - 4, 20 - x, y - 5, 23 - y) < 3
            mask[y, x] = (inside and edge) or (x == 39 and y < 2)
    np.testing.assert_array_equal(result[~mask], expected[~mask])
    assert np.all(result[mask, :3] == np.array([255, 0, 0]))
    if mode == "RGBA":
        np.testing.assert_array_equal(result[:, :, 3], original[:, :, 3])
    assert result.shape[:2] == original.shape[:2]
    assert sha256_file(source) == digest


def test_protect_source_and_output(image_path, tmp_path):
    with pytest.raises(ValueError, match="differ"):
        draw_boxes(image_path, image_path, [])
    with pytest.raises(ValueError, match="PNG"):
        draw_boxes(image_path, tmp_path / "lossy.jpg", [])
    with pytest.raises(ValueError, match="positive"):
        draw_boxes(image_path, tmp_path / "bad.png", [], width=0)
    with pytest.raises(ValueError, match="bbox_xyxy"):
        draw_boxes(image_path, tmp_path / "bad.png", [(0, 0, 65, 3)])
    output = tmp_path / "exists.png"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        draw_boxes(image_path, output, [])
    assert output.read_bytes() == b"preserve"


def test_no_implicit_16bit_normalization(tmp_path):
    path = tmp_path / "16.png"
    Image.fromarray(np.array([[0, 65535]], dtype=np.uint16)).save(path)
    with pytest.raises(ValueError, match="8-bit"):
        draw_boxes(path, tmp_path / "display.png", [])


@pytest.mark.parametrize("box", [[0, 0, True, 2], [0, 0, "2", 3], [0, 0, 2]])
def test_malformed_renderer_coordinates(image_path, tmp_path, box):
    with pytest.raises(ValueError, match="four finite"):
        draw_boxes(image_path, tmp_path / "bad.png", [box])
