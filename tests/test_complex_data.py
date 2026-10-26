"""Synthetic complex arrays test display preparation without SAR science claims."""

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

from sar_vlm.cli.inspect_dataset import inventory
from sar_vlm.cli.prepare_complex_preview import load_complex, prepare
from sar_vlm.data.reader import sha256_file


def test_complex_preview_preserves_source_and_geometry(tmp_path):
    path = tmp_path / "complex.npy"
    pixels = (np.arange(24).reshape(4, 6) + 1j).astype(np.complex64)
    np.save(path, pixels)
    before = sha256_file(path)
    output = tmp_path / "preview.png"
    trace = prepare(path, output)
    assert sha256_file(path) == before
    assert Image.open(output).size == (6, 4)
    assert trace["source_shape"] == [4, 6]
    assert trace["annotation_status"] == "pending"
    assert not trace["axes_changed"] and not trace["calibrated_backscatter"]
    assert output.with_suffix(".png.json").is_file()
    with pytest.raises(FileExistsError):
        prepare(path, output)


def test_requires_explicit_mat_variable(tmp_path):
    path = tmp_path / "complex.mat"
    data = np.array([[1 + 2j, 3 + 4j]])
    savemat(path, {"documented_variable": data})
    with pytest.raises(ValueError, match="requires --key"):
        load_complex(path)
    np.testing.assert_array_equal(load_complex(path, "documented_variable"), data)


@pytest.mark.parametrize(
    "data",
    [np.ones((2, 3)), np.ones((2, 3, 4), dtype=complex), np.array([[complex(float("nan"), 0)]])],
)
def test_rejects_unconfirmed_encodings(tmp_path, data):
    path = tmp_path / "input.npy"
    np.save(path, data)
    with pytest.raises(ValueError):
        prepare(path, tmp_path / "out.png")


def test_inventory_does_not_infer_format_or_labels(tmp_path):
    (tmp_path / "data.mat").write_bytes(b"not opened by inventory")
    (tmp_path / "label.xml").write_text("not interpreted")
    report = inventory(tmp_path)
    assert report["file_count"] == 2
    assert report["extensions"] == {".mat": 1, ".xml": 1}
    assert not report["training_data_ready"]
    assert report["complex_encoding"] == "UNCONFIRMED"


def test_npz_and_complex_tiff_require_explicit_data(tmp_path):
    import tifffile

    data = np.array([[1 + 3j, 4 + 2j]], dtype=np.complex64)
    npz = tmp_path / "example.npz"
    np.savez(npz, documented=data)
    with pytest.raises(ValueError, match="requires --key"):
        load_complex(npz)
    np.testing.assert_array_equal(load_complex(npz, "documented"), data)
    tif = tmp_path / "example.tif"
    tifffile.imwrite(tif, data)
    np.testing.assert_array_equal(load_complex(tif), data)
