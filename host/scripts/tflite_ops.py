"""Print the operators and tensor shapes of .tflite files (handy to check MicroFlow compatibility).

Usage: uv run python scripts/tflite_ops.py model.tflite [...]
"""

import sys

import tensorflow as tf


def shapes(tensors: dict, ids) -> list[tuple[int, ...]]:
    return [tuple(int(d) for d in tensors[i]["shape"]) for i in ids if i >= 0]


for path in sys.argv[1:]:
    interp = tf.lite.Interpreter(model_path=path)
    tensors = {t["index"]: t for t in interp.get_tensor_details()}
    print(path)
    for op in interp._get_ops_details():
        print(f"  {op['op_name']:<20} {shapes(tensors, op['inputs'])} -> {shapes(tensors, op['outputs'])}")
