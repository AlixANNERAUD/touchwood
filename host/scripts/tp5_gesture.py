"""TP 5, part 2 (the "Colab" side): produce the gesture model and the fixed test windows for the
`firmware/gesture` crate, and print the reference probabilities to compare with the board.

Default: reproduce the TP 2 notebook from the raw recordings in `host/data/tp2/` (same cleaning,
windows, chronological split, CNN and int8 conversion), then export:

    uv run python scripts/tp5_gesture.py

With a model already trained at TP 2 and the `test_windows.h` of the TP's Colab code (section 2.2):

    uv run python scripts/tp5_gesture.py --tflite model_int8.tflite --test-windows test_windows.h

Outputs (in `firmware/gesture/`): `models/gesture_model.tflite` (embedded by the firmware with
`include_bytes!`, instead of the `gesture_model.h` C array) and `src/test_windows.rs` (the TP's
`test_windows.h`, plus WINDOW and the class order, so the firmware cannot drift from them).

About the data: the TP says accelerometer in g at 100 Hz, but the values step by 0.061 = 2000/32768
(LSM9DS1 gyroscope at +/-2000 dps, `IMU.readGyroscope`) and change every 3.55 rows on average, so
the logger printed ~422 rows/s from a sensor running at 119 Hz. The firmware reproduces that.
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from touchwood.export import check_nanoglia, to_int8_tflite

HOST = Path(__file__).resolve().parents[1]
DATA = HOST / "data" / "tp2"
FIRMWARE = HOST.parent / "firmware" / "gesture"

# TP 2 notebook choices.
CLASSES = ["idle", "side", "forward", "up_down"]  # labels 0..3, in this order
WINDOW = 200
OVERLAP = 0.5
N_AXES = 3


def load_signals() -> dict[str, pd.DataFrame]:
    """As in the notebook: read each file, drop rows with missing values (one corrupted line in idle)."""
    signals = {}
    for name in CLASSES:
        df = pd.read_csv(DATA / f"{name}.txt", sep=",")
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0).reset_index(drop=True)
        signals[name] = df
        print(f"{name:8}: {len(df)} rows")
    return signals


def slice_windows(values: np.ndarray, win: int, overlap: float) -> np.ndarray:
    """Same windows as the notebook's `slice_signals`: step win - round(win * (1 - overlap))."""
    step = max(1, win if overlap == 0 else win - round(win * (1 - overlap)))
    starts = range(0, len(values) - win + 1, step)
    return np.stack([values[s : s + win] for s in starts]).astype(np.float32)


def split_dataset(signals: dict[str, pd.DataFrame]):
    """Chronological 70 / 10 / 20 % split of each recording, before windowing (no leakage)."""
    parts = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for label, name in enumerate(CLASSES):
        values = signals[name][["x", "y", "z"]].to_numpy()
        n = len(values)
        bounds = {"train": (0, int(0.7 * n)), "val": (int(0.7 * n), int(0.8 * n)), "test": (int(0.8 * n), n)}
        for part, (a, b) in bounds.items():
            w = slice_windows(values[a:b], WINDOW, OVERLAP)
            parts[part][0].append(w)
            parts[part][1].append(np.full(len(w), label))
    return {k: (np.concatenate(x)[..., None], np.concatenate(y)) for k, (x, y) in parts.items()}


def build_model(batch_size: int | None = None) -> tf.keras.Model:
    """The notebook's `build_1DCNN`: input (WINDOW, 3, 1), Conv2D over time only."""
    inp = layers.Input((WINDOW, N_AXES, 1), batch_size=batch_size)
    x = layers.Conv2D(16, (3, 1), activation="relu")(inp)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Conv2D(32, (3, 1), activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.MaxPooling2D((4, 1))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(len(CLASSES), activation="softmax")(x)
    return tf.keras.Model(inp, out)


def int8_predict(tflite: bytes, x: np.ndarray) -> np.ndarray:
    """Probabilities of an int8 model, inputs quantized like the TP's code (np.round, clip)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tf.lite.Interpreter deprecation notice
        interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    s, z = inp["quantization"]
    s_o, z_o = out["quantization"]
    probs = []
    for w in x:
        q = w if s == 0 else np.clip(np.round(w / s + z), -128, 127)
        interp.set_tensor(inp["index"], q.astype(inp["dtype"]).reshape(inp["shape"]))
        interp.invoke()
        raw = interp.get_tensor(out["index"]).astype(np.float32).reshape(-1)
        probs.append(raw if s_o == 0 else (raw - z_o) * s_o)
    return np.asarray(probs)


def confusion(y: np.ndarray, pred: np.ndarray) -> str:
    m = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for a, b in zip(y, pred):
        m[a, b] += 1
    head = "          " + " ".join(f"{c[:7]:>7}" for c in CLASSES)
    rows = [f"{CLASSES[i][:8]:>8}  " + " ".join(f"{v:7d}" for v in m[i]) for i in range(len(CLASSES))]
    return "\n".join([head, *rows])


def train_from_recordings() -> tuple[bytes, np.ndarray, np.ndarray]:
    tf.keras.utils.set_random_seed(0)
    data = split_dataset(load_signals())
    (x_tr, y_tr), (x_val, y_val), (x_test, y_test) = data["train"], data["val"], data["test"]
    print(f"windows: train {len(x_tr)}, val {len(x_val)}, test {len(x_test)}")

    model = build_model()
    model.compile(tf.keras.optimizers.Adam(0.001), "sparse_categorical_crossentropy", metrics=["accuracy"])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    model.fit(x_tr, y_tr, batch_size=32, epochs=50, validation_data=(x_val, y_val), callbacks=[stop], verbose=0)
    acc_float = (model.predict(x_test, verbose=0).argmax(1) == y_test).mean()

    # Export with a static batch size of 1 (otherwise Flatten emits SHAPE / PACK).
    export = build_model(batch_size=1)
    export.set_weights(model.get_weights())
    # TP 2 calibrated on X_tr[:200]: the first 200 windows, all idle since X_tr is in class order.
    tflite_tp2 = to_int8_tflite(export, x_tr[:200])
    # Same number of windows, drawn from all classes.
    calib = x_tr[np.random.default_rng(0).permutation(len(x_tr))[:200]]
    calib_tflite = to_int8_tflite(export, calib)

    # One int8 scale cannot fit both idle (~0.5 dps) and the gestures (up to ~200 dps): the
    # all-class calibration rounds idle to ~0. Pick the calibration on the validation set.
    print(f"test accuracy (float): {acc_float:.2%}")
    candidates = {"idle only, as TP 2": tflite_tp2, "all classes": calib_tflite}
    scores = {}
    for name, model_bytes in candidates.items():
        val = (int8_predict(model_bytes, x_val).argmax(1) == y_val).mean()
        test = (int8_predict(model_bytes, x_test).argmax(1) == y_test).mean()
        scale = input_scale(model_bytes)
        scores[name] = val
        print(f"int8, calibration {name:18}: input scale {scale:.3f} dps/step | val {val:.2%} | test {test:.2%}")
    best = max(scores, key=scores.get)
    tflite = candidates[best]
    pred = int8_predict(tflite, x_test).argmax(1)
    print(f"kept: calibration {best}\nint8 confusion matrix (rows: true, columns: predicted):\n" + confusion(y_test, pred))
    return tflite, x_test.reshape(len(x_test), -1), y_test


def input_scale(tflite: bytes) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tf.lite.Interpreter(model_content=tflite).get_input_details()[0]["quantization"][0]


def parse_test_windows_h(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the `test_labels` and `test_windows` arrays of the TP's test_windows.h."""
    text = path.read_text()
    labels = re.search(r"test_labels\[[^\]]*\]\s*=\s*\{([^}]*)\}", text)
    body = re.search(r"test_windows\[[^\]]*\]\[(\d+)\]\s*=\s*\{(.*)\};", text, re.S)
    if not labels or not body:
        raise ValueError(f"{path}: test_labels / test_windows not found")
    n_inputs = int(body.group(1))
    values = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", body.group(2))]
    return np.asarray(values, dtype=np.float32).reshape(-1, n_inputs), np.asarray(
        [int(v) for v in labels.group(1).split(",")]
    )


def write_test_windows_rs(path: Path, windows: np.ndarray, labels: np.ndarray) -> None:
    # repr() keeps a '.' or an exponent, so every value is a valid Rust float literal.
    rows = ",\n".join("    [" + ", ".join(repr(float(v)) for v in w) + "]" for w in windows)
    classes = ", ".join(f'"{c}"' for c in CLASSES)
    path.write_text(
        "// Generated by host/scripts/tp5_gesture.py: the TP 2 pre-processing parameters and one\n"
        "// pre-processed window per class (the TP's test_windows.h). Do not edit by hand.\n\n"
        f"/// Samples per window (rows of the recordings).\npub const WINDOW: usize = {WINDOW};\n"
        f"/// Class names, in the order of the model outputs.\npub const CLASSES: [&str; {len(CLASSES)}] = [{classes}];\n\n"
        f"pub const N_TEST: usize = {len(windows)};\n"
        f"pub const TEST_LABELS: [usize; N_TEST] = [{', '.join(str(int(c)) for c in labels)}];\n"
        f"pub static TEST_WINDOWS: [[f32; {windows.shape[1]}]; N_TEST] = [\n{rows},\n];\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tflite", type=Path, help="model trained at TP 2 (default: retrain from host/data/tp2)")
    ap.add_argument("--test-windows", type=Path, help="test_windows.h from the TP's Colab code (needed with --tflite)")
    args = ap.parse_args()

    if args.tflite:
        if not args.test_windows:
            ap.error("--tflite needs --test-windows (the windows must match the model's pre-processing)")
        tflite = args.tflite.read_bytes()
        windows, labels = parse_test_windows_h(args.test_windows)
    else:
        tflite, x_test, y_test = train_from_recordings()
        # As in the TP's code: the first test window of each class.
        idx = [int(np.where(y_test == c)[0][0]) for c in range(len(CLASSES))]
        windows, labels = x_test[idx], y_test[idx]
    if windows.shape[1] != WINDOW * N_AXES:
        raise SystemExit(f"test windows have {windows.shape[1]} values, expected WINDOW * 3 = {WINDOW * N_AXES}")

    print("ops:", sorted(check_nanoglia(tflite)))
    (FIRMWARE / "models").mkdir(parents=True, exist_ok=True)
    (FIRMWARE / "models" / "gesture_model.tflite").write_bytes(tflite)
    write_test_windows_rs(FIRMWARE / "src" / "test_windows.rs", windows, labels)
    print(f"gesture_model.tflite: {len(tflite)} bytes ({len(tflite) / 1024:.2f} KB)")

    probs = int8_predict(tflite, windows)
    print("reference probabilities (one line per window, compare with the board):")
    print("  " + " ".join(f"{c:>7}" for c in CLASSES))
    for k, (p, c) in enumerate(zip(probs, labels)):
        print(f"  window {k} | expected: {CLASSES[c]:8} | " + " ".join(f"{v:.3f}" for v in p))


if __name__ == "__main__":
    main()
