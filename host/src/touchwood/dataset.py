"""Datasets: real recordings -> tap windows, a synthetic stand-in, and the model input format.

Raw dataset format (a dict of numpy arrays, N taps):
    audio_<b>  (N, AUDIO_LEN) int16/float   microphone window of board b, starting PRE samples before the onset
    imu_<b>    (N, IMU_LEN, 6) float        IMU window of board b (optional)
    y          (N,) int                     zone index
    group      (N,) str                     recording session (for session-wise splits)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .model import AUDIO_LEN, IMU_AXES, IMU_LEN
from .taps import cut_windows, detect_onsets, pair_onsets

FS = 16_000
PRE = 80  # samples kept before the onset (5 ms)
LOGE_SCALE = 4.5  # log10(32768) ≈ 4.5, maps the log peak of an int16 window to ~[0, 1]


def prepare(raw: dict, boards: tuple[str, ...], use_imu: bool, imu_scale: float = 1.0) -> dict:
    """Raw dataset -> model inputs. The firmware must apply exactly the same transform.

    - Audio is normalised **per board** by its own peak, so each board can run its encoder alone
      (split computing). The peak itself is kept as a separate log-energy feature (`loge_b`):
      the head sees the energy *ratio* between boards, which a shared normalisation would
      otherwise need both boards' raw data to compute.
    - IMU keeps its physical scale (one global constant), since amplitude carries information.
    """
    inputs = {}
    for b in boards:
        a = raw[f"audio_{b}"].astype(np.float32)
        a = a - a.mean(axis=1, keepdims=True)
        peak = np.abs(a).max(axis=1, keepdims=True) + 1e-6
        inputs[f"audio_{b}"] = (a / peak)[:, None, :, None]
        inputs[f"loge_{b}"] = (np.log10(peak + 1.0) / LOGE_SCALE).astype(np.float32)
        if use_imu:
            inputs[f"imu_{b}"] = (raw[f"imu_{b}"].astype(np.float32) / imu_scale)[:, None, :, :]
    return inputs


def subset(d: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in d.items()}


# --------------------------------------------------------------------------- real recordings


def load_recordings(root: Path, tol_s: float = 0.03) -> tuple[dict, list[str]]:
    """Build a raw dataset from `scripts/record.py` outputs.

    Files: <root>/<session>/<board>_<label>_<time>.npz. When both boards recorded the same
    (session, label) at the same time, taps are paired across boards from their onset times
    (the offset between the two recordings is estimated automatically). Labels named `noise`
    are skipped. Returns (raw, sorted zone labels).
    """
    files = defaultdict(dict)
    for path in sorted(Path(root).glob("*/*.npz")):
        board, label = path.stem.split("_")[:2]
        if label != "noise":
            files[(path.parent.name, label)][board] = path
    boards = sorted({b for per_board in files.values() for b in per_board})
    labels = sorted({label for _, label in files})
    out = defaultdict(list)
    for (session, label), per_board in sorted(files.items()):
        if set(per_board) != set(boards):
            print(f"skip {session}/{label}: boards {sorted(per_board)} instead of {boards}")
            continue
        audio = {b: np.load(p)["audio"] for b, p in per_board.items()}
        onsets = {b: detect_onsets(x, FS) for b, x in audio.items()}
        if len(boards) == 2:
            pairs = pair_onsets(onsets[boards[0]], onsets[boards[1]], FS, tol_s)
            onsets = {boards[0]: pairs[:, 0], boards[1]: pairs[:, 1]}
        windows = {b: cut_windows(audio[b], onsets[b], AUDIO_LEN, PRE) for b in boards}
        n = min(len(w) for w in windows.values())  # windows too close to the edges are dropped
        for b in boards:
            out[f"audio_{b}"].append(windows[b][:n])
        out["y"].append(np.full(n, labels.index(label)))
        out["group"].append(np.full(n, session))
        print(f"{session}/{label}: {n} taps")
    if not out:
        raise FileNotFoundError(f"no recordings under {Path(root).resolve()}")
    return {k: np.concatenate(v) for k, v in out.items()}, labels


# --------------------------------------------------------------------------- synthetic data


def synthetic(
    n_per_zone: int = 100,
    grid: tuple[int, int] = (3, 3),
    table: tuple[float, float] = (1.2, 0.8),
    boards: dict[str, tuple[float, float]] | None = None,
    sessions: int = 2,
    seed: int = 0,
) -> tuple[dict, list[str]]:
    """Physics-flavoured stand-in to debug the pipeline before real data exists. NOT a result.

    Each tap excites a few plate modes whose weights depend on the tap position, propagates
    with a frequency-dependent speed and distance-dependent damping of high frequencies, and
    reflects off the four edges (image sources). The IMU sees a decaying low-frequency wobble
    whose gyro axes follow the direction of arrival. Sessions differ slightly (gain, noise,
    mode frequencies), which mimics day-to-day drift.
    """
    rng = np.random.default_rng(seed)
    lx, ly = table
    boards = boards or {"A": (0.05, ly / 2), "B": (lx - 0.05, ly / 2)}
    gx, gy = grid
    centers = [((i + 0.5) * lx / gx, (j + 0.5) * ly / gy) for j in range(gy) for i in range(gx)]
    labels = [f"z{k}" for k in range(len(centers))]
    modes = [(1, 1), (2, 1), (1, 2), (3, 1), (2, 2), (3, 2)]
    base_freqs = np.array([450, 800, 1250, 1900, 2600, 3500.0])
    t_audio = np.arange(AUDIO_LEN) / FS
    t_imu = np.arange(IMU_LEN) / 1000.0
    out = defaultdict(list)
    for s in range(sessions):
        freqs = base_freqs * rng.uniform(0.97, 1.03, len(base_freqs))
        gain, noise = rng.uniform(0.8, 1.2), rng.uniform(10, 25)
        for zone, (cx, cy) in enumerate(centers):
            for _ in range(n_per_zone):
                p = np.array([cx, cy]) + rng.uniform(-0.25, 0.25, 2) * [lx / gx, ly / gy]
                force = gain * rng.lognormal(0, 0.3)
                weights = np.array(
                    [abs(np.sin(m * np.pi * p[0] / lx) * np.sin(n * np.pi * p[1] / ly)) for m, n in modes]
                )
                images = [
                    (p, 1.0),
                    ((-p[0], p[1]), 0.5),
                    ((2 * lx - p[0], p[1]), 0.5),
                    ((p[0], -p[1]), 0.5),
                    ((p[0], 2 * ly - p[1]), 0.5),
                ]
                for b, pos in boards.items():
                    x = rng.normal(0, noise, AUDIO_LEN)
                    for src, refl in images:
                        d = np.hypot(*(np.asarray(src) - pos)) + 0.02
                        for f, w in zip(freqs, weights):
                            tt = t_audio - (PRE / FS + d / (300 + 0.1 * f))  # higher frequencies travel faster
                            env = np.where(tt >= 0, np.exp(-tt / (0.004 + 0.002 * rng.random())), 0.0)
                            amp = 4000 * force * refl * w / np.sqrt(d) * np.exp(-f * d / 4000)
                            x += amp * env * np.sin(2 * np.pi * f * tt)
                    out[f"audio_{b}"].append(np.clip(x, -32768, 32767).astype(np.int16))
                    d = np.hypot(*(p - pos)) + 0.05
                    u = (p - pos) / d
                    decay = np.exp(-np.maximum(t_imu - 0.01, 0) / 0.02) * (t_imu >= 0.01)
                    wobble = force / d * decay * np.sin(2 * np.pi * 45 * t_imu)
                    imu = np.stack(
                        [
                            0.1 * u[0] * wobble,
                            0.1 * u[1] * wobble,
                            wobble,
                            u[1] * wobble,
                            -u[0] * wobble,
                            0.05 * wobble,
                        ],
                        axis=1,
                    )
                    out[f"imu_{b}"].append(imu + rng.normal(0, 0.03, imu.shape))
                out["y"].append(zone)
                out["group"].append(f"S{s + 1}")
    raw = {k: np.asarray(v) for k, v in out.items()}
    assert raw["audio_A"].shape[1:] == (AUDIO_LEN,) and raw["imu_A"].shape[1:] == (IMU_LEN, IMU_AXES)
    return raw, labels
