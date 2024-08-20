"""Utility functions for filesystem and display options."""

from pathlib import Path

import cv2                      # type: ignore

import matplotlib.pyplot as plt

import numpy as np


__all__ = [
    "load_image",
    "display_image",
    "save_image",
]


def load_image(path: Path) -> np.ndarray:
    """Load an image using OpenCV.

    Args:
        path: a Path object to the image.

    Returns:
        np.ndarray: the image, unchanged.

    Raises:
        ValueError: if path points to an invalid image file.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("Invalid image file.")

    return image


def display_image(image: np.ndarray):
    """Display an image."""
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


def save_image(image: np.ndarray, path: Path):
    """Save an image."""
    cv2.imwrite(str(path), image)
