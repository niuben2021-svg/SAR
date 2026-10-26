"""Create a separate amplitude display PNG from explicitly selected complex SAR pixels."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from sar_vlm.data.reader import sha256_file
from sar_vlm.runtime import timestamp, write_json


def load_complex(path: Path, key: str | None = None) -> np.ndarray:
    """Read explicit NPY/NPZ/MAT v5/complex TIFF; never guess axes or MAT variables."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        pixels = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        if not key:
            raise ValueError("NPZ requires --key; array names are not inferred")
        with np.load(path, allow_pickle=False) as archive:
            pixels = archive[key]
    elif suffix == ".mat":
        if not key:
            raise ValueError("MAT requires --key from the dataset documentation")
        from scipy.io import loadmat

        pixels = loadmat(path, variable_names=[key])[key]
    elif suffix in {".tif", ".tiff"}:
        import tifffile

        pixels = tifffile.imread(path)
    else:
        raise ValueError("supported explicit complex arrays: .npy/.npz/MAT v5/.tif")
    if pixels.ndim != 2 or not np.iscomplexobj(pixels):
        raise ValueError(
            "expected a 2-D complex array; real/imag encodings and axes need documentation"
        )
    if pixels.size == 0 or not np.isfinite(pixels).all():
        raise ValueError("complex array must be nonempty and finite")
    return pixels


def prepare(
    source: Path, output: Path, key: str | None = None, lower: float = 1.0, upper: float = 99.5
) -> dict:
    """Write percentile-clipped 20log10(abs(z)) display; preserve source and pixel geometry.

    This is an uncalibrated display transform, not sigma0, phase analysis, ground
    truth generation, or a claim that a MAT variable contains the correct SAR channel.
    """
    if not 0 <= lower < upper <= 100:
        raise ValueError("require 0 <= lower percentile < upper percentile <= 100")
    sidecar = output.with_suffix(output.suffix + ".json")
    if output.resolve() == source.resolve() or output.suffix.lower() != ".png":
        raise ValueError("output must be a separate PNG")
    if output.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite preview or trace")
    digest = sha256_file(source)
    pixels = load_complex(source, key)
    amplitude = np.abs(pixels.astype(np.complex128))
    display_db = 20 * np.log10(np.maximum(amplitude, np.finfo(np.float64).tiny))
    lo, hi = (float(v) for v in np.percentile(display_db, [lower, upper]))
    normalized = (
        np.zeros_like(display_db) if hi == lo else np.clip((display_db - lo) / (hi - lo), 0, 1)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(normalized * 255).astype(np.uint8)).save(output)
    if sha256_file(source) != digest:
        raise RuntimeError("source file changed during preparation")
    result = {
        "timestamp": timestamp(),
        "source": str(source.resolve()),
        "source_sha256": digest,
        "source_dtype": str(pixels.dtype),
        "source_key": key,
        "source_shape": list(pixels.shape),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "display_transform": "20log10(abs(z)); percentile clip",
        "percentiles": [lower, upper],
        "display_db_bounds": [lo, hi],
        "calibrated_backscatter": False,
        "axes_changed": False,
        "complex_source_preserved": True,
        "annotation_status": "pending",
    }
    write_json(sidecar, result)
    return result


def main() -> None:
    """Convert a user-selected complex channel into a traceable model display copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key")
    parser.add_argument("--lower-percentile", type=float, default=1)
    parser.add_argument("--upper-percentile", type=float, default=99.5)
    args = parser.parse_args()
    prepare(args.input, args.output, args.key, args.lower_percentile, args.upper_percentile)
    print(
        f"Display PNG saved to {args.output}; labels/complex encoding still require dataset review"
    )


if __name__ == "__main__":
    main()
