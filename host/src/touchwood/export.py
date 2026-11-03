"""int8 TFLite export for MicroFlow, and a small runner to evaluate the exported models on the PC."""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import tensorflow as tf

MICROFLOW_OPS = {
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "FULLY_CONNECTED",
    "AVERAGE_POOL_2D",
    "SOFTMAX",
    "RESHAPE",
    "TRANSPOSE",
}


def to_int8_tflite(model, calib: np.ndarray, int8_io: bool = True) -> bytes:
    """Full-integer quantization, with the MicroFlow-specific options.

    `calib`: representative inputs (a few hundred), same shape as the model input minus the batch.
    """
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([calib[i : i + 1].astype("float32")] for i in range(len(calib)))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if int8_io:
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    # MicroFlow's FULLY_CONNECTED only supports per-tensor weight quantization.
    conv._experimental_disable_per_channel_quantization_for_dense_layers = True
    # The converter prints the whole SavedModel signature; keep notebooks readable.
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf.get_logger().setLevel("ERROR")
        return conv.convert()


def ops(tflite: bytes) -> set[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {o["op_name"] for o in tf.lite.Interpreter(model_content=tflite)._get_ops_details()}


def check_microflow(tflite: bytes) -> set[str]:
    """Raise if the model uses an operator MicroFlow does not implement."""
    used = ops(tflite)
    if used - MICROFLOW_OPS:
        raise ValueError(f"operators not supported by MicroFlow: {sorted(used - MICROFLOW_OPS)}")
    return used


class TFLiteModel:
    """Run an exported (int8 in/out) model on float inputs, one sample at a time like the MCU."""

    def __init__(self, tflite: bytes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # tf.lite.Interpreter deprecation notice
            self.interp = tf.lite.Interpreter(model_content=tflite)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        in_scale, in_zp = self.inp["quantization"]
        out_scale, out_zp = self.out["quantization"]
        results = []
        for sample in x:
            q = sample[None]
            if self.inp["dtype"] == np.int8:
                q = np.clip(np.round(q / in_scale + in_zp), -128, 127).astype(np.int8)
            self.interp.set_tensor(self.inp["index"], q)
            self.interp.invoke()
            y = self.interp.get_tensor(self.out["index"])[0]
            results.append((y.astype(np.float32) - out_zp) * out_scale if self.out["dtype"] == np.int8 else y)
        return np.stack(results)
