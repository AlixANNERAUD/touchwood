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
| `src/touchwood/taps.py` | Energy-based tap onset detection, windowing, and pairing of the same taps seen by two boards |
| `src/touchwood/dataset.py` | Recordings → tap dataset, synthetic stand-in, and `prepare()`, the input normalisation the firmware must reproduce |
| `src/touchwood/model.py` | Shared encoders + fusion head (MicroFlow-compatible), branch dropout, export helpers |
| `src/touchwood/export.py` | int8 TFLite conversion with the MicroFlow options, op-set check, and a PC runner for int8 models |
| `scripts/record.py` | Guided recording from one or two boards at once: random zone order, tap counting, quality checks, session log (`takes.csv`) |
| `scripts/export_microflow.py` | Keras → int8 `.tflite` that MicroFlow accepts, written to `firmware/inference/models/` |
| `scripts/tflite_ops.py` | List the operators and shapes of `.tflite` files |
| `notebooks/01_feasibility.ipynb` | Week-1 go / no-go: tap segmentation, plots, k-NN / random forest per zone |
| `notebooks/02_training.ipynb` | Joint training of the fusion model, ablations, frozen vs. unfrozen calibration, int8 export of the three parts |

## Recording a dataset

Follow **[DATASET.md](DATASET.md)**: setup, tapping protocol, session plan, quality checks. In short:

```sh
uv run python scripts/record.py --session 20261110_desk1_s1 --ports A=/dev/ttyACM0,B=/dev/ttyACM1 \
    --zones 9 --taps 25 --condition finger --operator alix
```

## TensorFlow notes

`tensorflow-model-optimization` (pruning / clustering, TP3) needs legacy Keras. Set this **before** importing TensorFlow:

```python
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
```
