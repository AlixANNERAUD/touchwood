"""Tap segmentation: find tap onsets in a continuous recording and cut fixed windows."""

from __future__ import annotations

import numpy as np


def frame_energy(x: np.ndarray, hop: int = 32) -> np.ndarray:
    """Short-time energy per hop (DC removed)."""
    x = x.astype(np.float32) - np.mean(x)
    n = len(x) // hop
    return (x[: n * hop].reshape(n, hop) ** 2).mean(axis=1)


def detect_onsets(
    x: np.ndarray,
    fs: int = 16_000,
    hop: int = 32,
    threshold_db: float = 15.0,
    refractory_s: float = 0.25,
) -> np.ndarray:
    """Sample indices where the energy rises `threshold_db` above the median noise floor.

    Same idea as the planned on-device trigger: cheap, always-on, energy-based.
    """
    e_db = 10 * np.log10(frame_energy(x, hop) + 1e-9)
    floor = np.median(e_db)
    above = e_db > floor + threshold_db
    onsets, last = [], -np.inf
    for i in np.flatnonzero(above & ~np.r_[False, above[:-1]]):
        t = i * hop
        if t - last >= refractory_s * fs:
            onsets.append(t)
            last = t
    return np.asarray(onsets, dtype=int)


def cut_windows(x: np.ndarray, onsets: np.ndarray, length: int = 800, pre: int = 80) -> np.ndarray:
    """Windows of `length` samples starting `pre` samples before each onset (edges skipped)."""
    out = [x[o - pre : o - pre + length] for o in onsets if o - pre >= 0 and o - pre + length <= len(x)]
    return np.stack(out) if out else np.empty((0, length), dtype=x.dtype)
