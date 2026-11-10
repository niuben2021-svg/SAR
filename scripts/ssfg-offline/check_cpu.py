"""Exercise required scientific operations without claiming GPU validation."""
import tempfile
from pathlib import Path
import cv2
import numpy as np
import scipy
from scipy.io import savemat, loadmat
from scipy.signal.windows import taylor
import h5py
from PIL import Image

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    assert np.isfinite(np.fft.fft2(values)).all()
    _, _, directions = np.linalg.svd(values)
    assert directions.shape == (4, 4)
    savemat(root / 'test.mat', {'values': values})
    assert np.array_equal(loadmat(root / 'test.mat')['values'], values)
    assert taylor(16, 4, 30).shape == (16,)
    with h5py.File(root / 'test.h5', 'w') as handle:
        handle['values'] = values
    with h5py.File(root / 'test.h5') as handle:
        assert np.array_equal(handle['values'][:], values)
    Image.fromarray(values.astype(np.uint8)).save(root / 'test.png')
    assert np.array_equal(np.array(Image.open(root / 'test.png')), values)
    rectangle = ((0., 0.), (5., 5.), 0.)
    assert cv2.rotatedRectangleIntersection(rectangle, rectangle)[0] == 2
    assert cv2.minAreaRect(np.array([[0., 0.], [5., 0.], [5., 5.], [0., 5.]], dtype=np.float32))[1] == (5., 5.)
print('PASS: NumPy FFT/SVD, SciPy Taylor/MAT v5, h5py HDF5, Pillow PNG, OpenCV rotated geometry')
print('Versions:', np.__version__, scipy.__version__, h5py.__version__, cv2.__version__)
print('GPU / PyTorch validation: NOT RUN; run verify_train_env.py on internal training server')
