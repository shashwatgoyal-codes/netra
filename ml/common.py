"""Shared constants and preprocessing.

The browser app re-implements `crop_retina` + resize + normalize in
web/lib/preprocess.ts. Keep the two in sync: any change here must be mirrored
there, or the on-device model will see different inputs than it was trained on.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
# NETRA_DATA / NETRA_ARTIFACTS let a smoke test run the pipeline on throwaway data.
DATA = Path(os.environ.get("NETRA_DATA", ROOT / "data"))
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits.csv"
ARTIFACTS = Path(os.environ.get("NETRA_ARTIFACTS", ROOT / "artifacts"))

GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
# International clinical guidance: moderate NPDR or worse is "referable".
REFERABLE_FROM = 2

CACHE_SIZE = 416  # side of the square images stored in data/processed
INPUT_SIZE = 384  # side of the square tensor the model sees
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DARK_THRESHOLD = 10  # grayscale level below which a pixel counts as background


def crop_retina(img: Image.Image) -> Image.Image:
    """Crop the black border around the fundus and pad the result to a square."""
    rgb = np.asarray(img.convert("RGB"))
    gray = rgb.mean(axis=2)
    ys, xs = np.nonzero(gray > DARK_THRESHOLD)
    if len(xs) == 0:
        cropped = rgb
    else:
        cropped = rgb[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = cropped.shape[:2]
    side = max(h, w)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    square[top : top + h, left : left + w] = cropped
    return Image.fromarray(square)


def to_tensor_array(img: Image.Image, size: int = INPUT_SIZE) -> np.ndarray:
    """Square RGB image -> normalized float32 CHW array, as the browser does it."""
    resized = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return arr.transpose(2, 0, 1)
