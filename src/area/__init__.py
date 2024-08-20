"""Stuff."""

from pathlib import Path
from typing import Callable, Optional
from warnings import warn

from area import stats, util

import cv2  # type: ignore

import numpy as np

from toolz.functoolz import compose_left  # type: ignore

__all__ = [
    "percentile_threshold",
    "gaussian_blur",
    "kmeans_cluster",
    "contrast_adaptive_threshold",
    "util",
    "stats",
]

KMEANS_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
KMEANS_FLAGS = cv2.KMEANS_RANDOM_CENTERS
CONTRAST_THRESHOLD = 33


def percentile_threshold(
        image: np.ndarray,
        percentile: float = 90,
        mask: bool = True,
        **kwargs,
) -> np.ndarray:
    """Threshold an image according to a percentile.

    Change all pixels of an image that fall below a threshold represented by
    percentile to 0. Optionally (if mask), change those that fall above that
    threshold to the maximum allowed by the given dtype.

    Args:
        image: A two-dimensional array representing a single-channel image.
        percentile: The percentile at which to perform the
          thresholding. Defaults to 90.
        mask: Whether or not to maximize pixels above the threshold.

    Returns:
        np.ndarray: A two-dimensional array representing the single-channel
        processed image.

    """
    original_dtype = image.dtype
    maxval = np.iinfo(original_dtype).max

    thresh_image = np.where(
        image >= np.percentile(image, percentile),
        maxval if mask else image,
        0,
    ).astype(original_dtype)

    return thresh_image


def gaussian_blur(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian blur an image for pre-processing.

    Args:
        img: the image array.
        sigma: the value of sigma for the gaussian distribution, defaults to 2.
    """
    return cv2.GaussianBlur(img, (0, 0), sigma)


def kmeans_cluster(img: np.ndarray) -> np.ndarray:
    """Group a single-channel image into two clusters.

    Generate a binary image that can be used as a mask.
    """
    z = np.float32(img.ravel())
    column = z.reshape(z.shape[0], 1)  # type:ignore
    _, labels, centers = cv2.kmeans(
        column,
        2,
        None,
        KMEANS_CRITERIA,
        10,
        KMEANS_FLAGS,
    )

    dt = img.dtype
    centers_converted = centers.astype(dt)
    res = centers_converted[labels.flatten()].reshape(img.shape)

    info = np.iinfo(dt)
    mask = np.where(res > res.min(), info.max, info.min).astype(dt)

    return mask


def contrast_adaptive_threshold(
        img: np.ndarray,
        contrast_threshold: float = CONTRAST_THRESHOLD,
) -> np.ndarray:
    """Do stuff."""
    contrast = img.std()

    if contrast < contrast_threshold:
        segment_method = percentile_threshold
    else:
        segment_method = kmeans_cluster         # type: ignore

    return segment_method(gaussian_blur(img))   # type: ignore


def select_hoechst(p: Path) -> tuple[Path, list[Path]]:
    """Select from a list of paths which is the HOECHST sample."""
    ps = list(p.iterdir())
    p = next(filter(lambda p: "hoechst" in p.name.lower(), ps))
    ps.remove(p)
    return p, ps


def validate_mask(img: np.ndarray):
    """Ensure that img is in mask format.

    I.e. the only two values are 0 and the dtype maximum.

    Args:
        img: image to validate.

    Raises:
        ValueError: if any images contain invalid values.
    """
    maxval = np.iinfo(img.dtype).max
    if any(v not in (0, maxval) for v in np.unique(img)):
        raise ValueError("Invalid mask image.")


def validate_images(*imgs):
    """Ensure that all imgs share the same shape and dtype.

    Args:
        *imgs: images to validate.

    Raises:
        ValueError: if either the shape or dtype are mismatched.
    """
    shapes = [img.shape for img in imgs]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError("Mismatched shapes.")

    dtypes = [img.dtype for img in imgs]
    if not all(dtype == dtypes[0] for dtype in dtypes):
        raise ValueError("Mismatched dtypes.")


def percentage_area(part_img: np.ndarray, whole_img: np.ndarray) -> float:
    """Compute the proportion of total area represented by part_img.

    Divides pixels above 0 in part_img by pixels above 0 in whole_img.  Assumes
    that the images are binary, i.e. every pixel is either 0 or the maximum for
    the dtype.

    Args:
        part_img: the image representing part of the area.
        whole_img: the image representing the whole area.

    Returns:
        float: the proportion of the area of whole_img taken up by part_img.
    """
    validate_images(part_img, whole_img)
    for img in [part_img, whole_img]:
        validate_mask(img)

    part_unique, part_counts = np.unique(part_img, return_counts=True)
    whole_unique, whole_counts = np.unique(whole_img, return_counts=True)

    proportion = part_counts[1] / whole_counts[1]

    if not 0 <= proportion <= 1:
        warn(f"invalid proportion {proportion}")

    return proportion


def get_percentage_area(
        p: Path,
        denoise_method: Callable[[np.ndarray], np.ndarray] = gaussian_blur,
        segment_method: Callable[[np.ndarray], np.ndarray] = kmeans_cluster,
        whole_method: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        **kwargs,
) -> dict[Path, float]:
    """Calculate the percentage area of different cell types.

    Take the Hoechst image as the total area and calculate the percentage
    thereof occupied by the other images.

    Args:
        p: the path to the organoid.
        denoise_method: a function that denoises an image. Must take as a
            single argument the image to be denoised.
        segment_method: a function that segments an image. Must take as a
            single argument the image to be segmented.
        **kwargs: passed to the threshold and denoise processes.
    """
    h, ps = select_hoechst(p)

    if whole_method:
        proc = compose_left(util.load_image, whole_method)
    else:
        proc = compose_left(util.load_image, denoise_method, segment_method)

    h_thresh = proc(h)

    return {p: percentage_area(proc(p), h_thresh) for p in ps}
