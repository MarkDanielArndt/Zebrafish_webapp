import numpy as np
import pytest


@pytest.fixture
def synthetic_fish_image():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    import cv2
    cv2.ellipse(img, (128, 128), (100, 30), 0, 0, 360, (200, 200, 200), -1)
    return img


@pytest.fixture
def synthetic_fish_mask():
    mask = np.zeros((256, 256), dtype=np.uint8)
    import cv2
    cv2.ellipse(mask, (128, 128), (100, 30), 0, 0, 360, 255, -1)
    return mask
