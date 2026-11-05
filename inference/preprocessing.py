"""Conservative frame preparation with original-coordinate offsets."""

from __future__ import annotations

import numpy as np


def crop_black_padding(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Remove outer near-black padding, returning image and original x/y offset.

    A whole removed row/column must be near black. Skip nearly black frames
    and tiny inputs so a small light in a dark scene cannot drive a large crop.
    Callers must add the offsets to predictions before using camera geometry.
    """
    if image.ndim != 3 or image.shape[2] != 3 or not image.size:
        return image, 0, 0
    content = np.any(image > 12, axis=2)
    ys, xs = np.where(content)
    if not len(xs):
        return image, 0, 0
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    height, width = image.shape[:2]
    if (
        x1 - x0 < 32
        or y1 - y0 < 32
        or (x1 - x0) * (y1 - y0) < height * width * 0.1
        or (x0 == 0 and y0 == 0 and x1 == width and y1 == height)
    ):
        return image, 0, 0
    return image[y0:y1, x0:x1].copy(), x0, y0
