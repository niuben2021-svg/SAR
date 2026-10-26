"""Test native pixels, byte integrity and missing metadata without model imports."""

import numpy as np
import pytest
from PIL import Image

from sar_vlm.data.reader import SARReader, sha256_file


def test_readonly_8bit(image_path):
    before = sha256_file(image_path)
    record = SARReader().read(image_path)
    assert (record.width, record.height, record.channels, record.dtype) == (64, 48, 1, "uint8")
    assert record.intensity_range == (0, 255)
    assert record.display_transform == "none"
    assert record.metadata.incidence_angle_deg is None
    assert not record.metadata_valid["incidence_angle_deg"]
    with pytest.raises(ValueError):
        record.pixels[0, 0] = 0
    assert before == sha256_file(image_path) == record.sha256


@pytest.mark.parametrize("extension", ["png", "tiff"])
def test_preserves_16bit(tmp_path, extension):
    pixels = np.array([[0, 1024], [32768, 65535]], dtype=np.uint16)
    path = tmp_path / f"native16.{extension}"
    Image.fromarray(pixels).save(path)
    record = SARReader().read(path)
    assert record.dtype == "uint16"
    np.testing.assert_array_equal(record.pixels, pixels)
    assert record.intensity_range == (0, 65535)


def test_missing_and_bad_images(tmp_path):
    with pytest.raises(FileNotFoundError):
        SARReader().read(tmp_path / "absent.png")
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    with pytest.raises(ValueError, match="cannot decode"):
        SARReader().read(bad)


def test_palette_rejected_without_silent_conversion(tmp_path):
    path = tmp_path / "palette.png"
    Image.new("P", (4, 4)).save(path)
    with pytest.raises(ValueError, match="unsupported pixel mode"):
        SARReader().read(path)
