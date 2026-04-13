"""Model input preprocessing for grayscale eye crops."""

from __future__ import annotations

import numpy as np
import cv2

from drowsiness.config import IMG_SIZE


def preprocess_eye(eye: np.ndarray) -> np.ndarray:
    """
    eye: 2D grayscale (H, W) or single-channel (H, W, 1).
    Returns batch tensor (1, IMG_SIZE, IMG_SIZE, 1) float32 in [0, 1].
    """
    if eye.ndim == 3:
        eye = eye[:, :, 0]
    eye = cv2.resize(eye, (IMG_SIZE, IMG_SIZE))
    eye = eye.astype("float32") / 255.0
    eye = np.expand_dims(eye, axis=-1)
    eye = np.expand_dims(eye, axis=0)
    return eye
