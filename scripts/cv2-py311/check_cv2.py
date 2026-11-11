"""Exercise the requested cv2 build using in-memory engineering fixtures."""
import importlib.metadata
import cv2
import numpy as np

assert importlib.metadata.version('opencv-python-headless') == '4.11.0.86'
assert np.__version__ == '1.26.4'
image = np.zeros((64, 64, 3), dtype=np.uint8)
cv2.rectangle(image, (8, 8), (40, 40), (0, 255, 0), -1)
ok, encoded = cv2.imencode('.png', image)
assert ok
assert np.array_equal(cv2.imdecode(encoded, cv2.IMREAD_COLOR), image)
assert cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).shape == (64, 64)
assert cv2.resize(image, (32, 32)).shape == (32, 32, 3)
rectangle = ((20., 20.), (10., 20.), 30.)
intersection, points = cv2.rotatedRectangleIntersection(rectangle, rectangle)
assert intersection == 2 and points is not None
assert np.isclose(cv2.contourArea(cv2.boxPoints(rectangle)), 200., atol=1e-3)
assert np.isfinite(np.fft.fft2(image[:, :, 1])).all()
print('PASS: cv2', cv2.__version__, 'opencv-python-headless', importlib.metadata.version('opencv-python-headless'), 'NumPy', np.__version__)
print('PASS: PNG encode/decode, grayscale, resize, rotated geometry, NumPy FFT')
