"""Export a Keras model as an int8 .tflite that MicroFlow accepts (and check its operators).

The MicroFlow constraints are listed in the root README (§7); `touchwood.model` builds models
that respect them, and `touchwood.export` holds the conversion code shared with the notebooks.

Usage:
    uv run python scripts/export_microflow.py                  # placeholder random model
    uv run python scripts/export_microflow.py --keras model.keras --data X_calib.npy
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from touchwood.export import check_microflow, to_int8_tflite

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "firmware" / "inference" / "models" / "tap.tflite"


def placeholder_model(window: int = 400, n_zones: int = 9) -> tf.keras.Model:
    """Untrained stand-in with the same op set as the planned audio branch."""
    init = "random_normal"  # non-zero biases: an all-zero bias gets dropped by the converter
    inp = tf.keras.Input((1, window, 1), batch_size=1)
    x = tf.keras.layers.Conv2D(8, (1, 9), strides=(1, 2), activation="relu", bias_initializer=init)(inp)
    x = tf.keras.layers.Conv2D(16, (1, 5), strides=(1, 2), activation="relu", bias_initializer=init)(x)
    x = tf.keras.layers.AveragePooling2D((1, x.shape[2]))(x)
    x = tf.keras.layers.Reshape((16,))(x)
    out = tf.keras.layers.Dense(n_zones, activation="softmax", bias_initializer=init)(x)
    return tf.keras.Model(inp, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keras", type=Path, help="trained Keras model (.keras); default: placeholder")
    ap.add_argument("--data", type=Path, help="calibration inputs (.npy, shape (N, ...) matching the model input)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(0)  # reproducible placeholder / calibration

    model = tf.keras.models.load_model(args.keras) if args.keras else placeholder_model()
    calib = np.load(args.data) if args.data else np.random.randn(64, *model.input_shape[1:])
    tflite = to_int8_tflite(model, calib)
    print("ops:", sorted(check_microflow(tflite)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(tflite)
    print(f"params={model.count_params()} size={len(tflite)} B -> {args.out}")


if __name__ == "__main__":
    main()
