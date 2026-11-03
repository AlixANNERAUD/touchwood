"""Build (or load) a Keras model and export it as an int8 .tflite that MicroFlow accepts.

MicroFlow constraints (learned the hard way):
- operators: CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, AVERAGE_POOL_2D, SOFTMAX,
  RESHAPE, TRANSPOSE only;
- no Conv1D (lowers to EXPAND_DIMS/SQUEEZE): use Conv2D with a (1, k) kernel on (1, T, C);
- no MaxPool, no GlobalAveragePooling (MEAN): use AveragePooling2D over the full width;
- no Concatenate / Add: fuse branches in Rust, one model per branch;
- static batch size 1, otherwise Flatten emits SHAPE / PACK;
- dense layers must be quantized per-tensor;
- every layer needs a bias, and an all-zero bias gets dropped by the converter
  (fine once trained; keep use_bias=True);
- activations: ReLU / ReLU6 only (BatchNorm is fine, it is folded into the conv).

Usage:
    uv run python scripts/export_microflow.py                  # placeholder random model
    uv run python scripts/export_microflow.py --keras model.keras --data X_calib.npy
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "firmware" / "inference" / "models" / "tap.tflite"


def placeholder_model(window: int = 400, n_zones: int = 9) -> tf.keras.Model:
    """Untrained stand-in with the same op set as the planned audio branch."""
    init = "random_normal"  # non-zero biases, see constraints above
    inp = tf.keras.Input((1, window, 1), batch_size=1)
    x = tf.keras.layers.Conv2D(8, (1, 9), strides=(1, 2), activation="relu", bias_initializer=init)(inp)
    x = tf.keras.layers.Conv2D(16, (1, 5), strides=(1, 2), activation="relu", bias_initializer=init)(x)
    x = tf.keras.layers.AveragePooling2D((1, x.shape[2]))(x)
    x = tf.keras.layers.Reshape((16,))(x)
    out = tf.keras.layers.Dense(n_zones, activation="softmax", bias_initializer=init)(x)
    return tf.keras.Model(inp, out)


def to_int8_tflite(model: tf.keras.Model, calib: np.ndarray) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([calib[i : i + 1].astype("float32")] for i in range(len(calib)))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    conv._experimental_disable_per_channel_quantization_for_dense_layers = True
    return conv.convert()


SUPPORTED = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "AVERAGE_POOL_2D", "SOFTMAX", "RESHAPE", "TRANSPOSE"}


def check_ops(tflite: bytes) -> None:
    ops = {o["op_name"] for o in tf.lite.Interpreter(model_content=tflite)._get_ops_details()}
    unsupported = ops - SUPPORTED
    print("ops:", sorted(ops))
    if unsupported:
        raise SystemExit(f"operators not supported by MicroFlow: {sorted(unsupported)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keras", type=Path, help="trained Keras model (.keras); default: placeholder")
    ap.add_argument("--data", type=Path, help="calibration inputs (.npy, shape (N, 1, T, C))")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(0)  # reproducible placeholder / calibration

    model = tf.keras.models.load_model(args.keras) if args.keras else placeholder_model()
    calib = np.load(args.data) if args.data else np.random.randn(64, *model.input_shape[1:])
    tflite = to_int8_tflite(model, calib)
    check_ops(tflite)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(tflite)
    print(f"params={model.count_params()} size={len(tflite)} B -> {args.out}")


if __name__ == "__main__":
    main()
