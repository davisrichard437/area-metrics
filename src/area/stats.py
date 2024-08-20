"""Stuff."""

from pathlib import Path

from area.util import load_image

import matplotlib.pyplot as plt

import numpy as np


def remove_zero_bins(
        counts: np.ndarray,
        bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove bins from histogram with zero counts for better plotting."""
    zero_indices = np.where(counts == 0)[0]
    clean_counts = np.delete(counts, zero_indices)
    clean_bins = np.delete(bins, zero_indices)
    return clean_counts, clean_bins


def img_hist(img: np.ndarray, p: Path, title: str):
    """Save histogram and cumulative curve for img to p."""
    counts, bins = np.histogram(img, bins="auto")
    clean_counts, clean_bins = remove_zero_bins(counts, bins)
    cumulative = np.cumsum(clean_counts)

    fig, ax1 = plt.subplots()
    ax1.set_title(title)
    fig.set_size_inches(10, 10)

    color = "tab:blue"
    ax1.set_xlabel("pixel intensity")
    ax1.set_ylabel("pixels", color=color)
    ax1.stairs(counts, bins, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()           # haha twinx
    color = "tab:red"
    ax2.set_ylabel("cumulative pixels", color=color)
    ax2.plot(clean_bins[:-1], cumulative, color=color)
    ax2.tick_params(axis="y", labelcolor=color)


def organoid_hists(organoid_dir: Path, out_dir: Path):
    """Save all histograms for an organoid_dir to out_dir."""
    imgs = {p: load_image(p) for p in organoid_dir.iterdir()}

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    for p, img in imgs.items():
        img_hist(img, out_dir / f"{p.stem}.png", p.name)
