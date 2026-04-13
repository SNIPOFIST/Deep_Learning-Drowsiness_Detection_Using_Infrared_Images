"""Face detection and eye region cropping using dlib."""

from __future__ import annotations

import cv2
import dlib
import numpy as np

from drowsiness.config import SHAPE_PREDICTOR_PATH

detector = dlib.get_frontal_face_detector()

if not SHAPE_PREDICTOR_PATH.is_file():
    raise FileNotFoundError(
        f"Missing dlib landmark file: {SHAPE_PREDICTOR_PATH}\n"
        "Download shape_predictor_68_face_landmarks.dat and place it in assets/."
    )

predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH))

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))


def crop_eye(gray: np.ndarray, landmarks, eye_points: list[int]):
    coords = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in eye_points])
    x, y, w, h = cv2.boundingRect(coords)
    return gray[y : y + h, x : x + w], (x, y, w, h)


def extract_eyes_from_frame(frame: np.ndarray):
    """
    BGR frame -> (left_eye_gray, right_eye_gray, left_box, right_box) or Nones if no face.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None, None, None, None

    face = faces[0]
    landmarks = predictor(gray, face)

    left_eye, left_box = crop_eye(gray, landmarks, LEFT_EYE)
    right_eye, right_box = crop_eye(gray, landmarks, RIGHT_EYE)
    return left_eye, right_eye, left_box, right_box
