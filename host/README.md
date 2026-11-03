# host

PC-side tools: recording, analysis, training and export. This is a [uv](https://docs.astral.sh/uv/) project on Python 3.13 (see the root README §8 for why not 3.14+).

```sh
uv sync --extra cpu          # or --extra cuda / --extra mps: PyTorch flavour, pick exactly one
uv run jupyter lab           # notebooks use the .venv kernel
uv run ruff check .
```

| Path | What |
|---|---|
| `src/touchwood/protocol.py` | Decoder for the `acquisition` firmware frames (resyncs on garbage) |
| `src/touchwood/taps.py` | Energy-based tap onset detection and windowing (the same idea as the planned on-device trigger) |
| `scripts/record.py` | Record a labelled session from a board → `data/raw/<session>/<board>_<label>_<time>.npz` |
| `scripts/export_microflow.py` | Keras → int8 `.tflite` that MicroFlow accepts, written to `firmware/inference/models/` |
| `scripts/tflite_ops.py` | List the operators and shapes of `.tflite` files |
| `notebooks/01_feasibility.ipynb` | Week-1 go / no-go: tap segmentation, plots, k-NN / random forest per zone |

## Recording a dataset

```sh
# acquisition firmware flashed (cd ../firmware && cargo run --release -p acquisition)
uv run python scripts/record.py --label z0 --seconds 30     # tap ~30 times on zone 0
uv run python scripts/record.py --label z1 --seconds 30
...
uv run python scripts/record.py --label noise --seconds 30  # no taps: background / negatives
# second board: --board B --port /dev/ttyACM1
```

## TensorFlow notes

`tensorflow-model-optimization` (pruning / clustering, TP3) needs legacy Keras. Set this **before** importing TensorFlow:

```python
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
```
