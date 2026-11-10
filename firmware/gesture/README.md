# gesture: TP 5 with Rust and nanoglia

TP 5 deploys the TP 2 gesture classifier on the Nano 33 BLE Sense. The TP uses Arduino IDE + MicroTFLite;
this crate does the same with Embassy and [nanoglia](https://github.com/AlixANNERAUD/nanoglia), to test the
whole Rust stack on a real model.

| TP (Arduino / MicroTFLite) | Here |
|---|---|
| `gesture_model.h` (C array, `alignas(16)`) | `models/gesture_model.tflite`, embedded with `include_bytes!` in a 16-byte aligned static |
| `test_windows.h` | `src/test_windows.rs`, which also carries `WINDOW` and the class order |
| Colab code of §2.1 and §2.2 | `host/scripts/tp5_gesture.py` (also prints the reference probabilities) |
| `tensorArena[kTensorArenaSize]` | `static mut MEMORY` given to a nanoglia `Arena` (18 KB) |
| `ModelInit` | `Graph::load_in` + `init_execution_context_in` |
| `ModelPrint*` | printed on connection: operators and kernels, input / output dims, type, scale, zero point, arena used |
| `ModelSetInput(x, i)` | `classify()`: quantizes the float window with the input scale and zero point (rounding like `np.round`) |
| `ModelRunInference` | `ctx.compute()` |
| `ModelGetOutput(j)` | `classify()`: dequantized probabilities |
| `Arduino_LSM9DS1` | `mod lsm9ds1`: gyroscope set up like `IMU.begin()` (119 Hz, ±2000 dps), values in dps |
| `micros()` sampling loop | `Ticker`, one row every 2.37 ms (see below) |

## Data: gyroscope, ~422 rows/s

The recordings (`host/data/tp2/*.txt`) are **not** what TP 2 and TP 5 describe (accelerometer in g at 100 Hz):

- values move in steps of **0.061 = 2000 / 32768**: the LSM9DS1 **gyroscope** at ±2000 dps (`IMU.readGyroscope`).
  The accelerometer at ±4 g would step by 0.00012 g, and at rest it would read ~1 g (gravity); `idle` averages
  (0.05, −0.38, 0.55);
- each value is repeated **3 or 4 rows** (3.55 on average): the logger printed ~422 rows/s from a sensor at 119 Hz
  (100 Hz rows would need a 28 Hz sensor, which the LSM9DS1 does not offer). One gesture cycle is ~500 rows (~1.2 s).

So a "200-sample, 2 s" window of TP 2 is really ~0.47 s. To feed the model what it was trained on, the firmware
reads the gyroscope output every 2.37 ms, new measurement or not, like the logger did.

## Model

`tp5_gesture.py` retrains the TP 2 notebook as written, from the four `.txt` files: rows with missing values
dropped (one corrupted line in `idle`), windows of 200 rows with 50 % overlap, chronological 70 / 10 / 20 % split
of each recording (1371 / 190 / 387 windows), classes `idle, side, forward, up_down`, the notebook's CNN
(Conv2D 3×1 ×2, max pooling, dense 32), int8 with int8 input / output. Seeded, so the numbers below are
reproducible; they are close to the notebook's (float 95.4 %, int8 92.0 %), not identical.

| | Accuracy (test) | Size |
|---|---|---|
| float (Keras) | 96.4 % | |
| int8, calibrated on `X_tr[:200]` as in TP 2 (idle windows only) — **kept** | 89.4 % (val 92.1 %) | 79.9 KB |
| int8, calibrated on 200 windows of all classes | 84.0 % (val 82.1 %) | |

The calibration is chosen on the validation set. Int8 confusion matrix of the kept model (rows: true):

```
             idle    side forward up_down
    idle       98       0       0       0
    side        0      96       0       0
 forward        0      39      55       2
 up_down        0       0       0      97
```

**Known weakness: idle is below one int8 step.** At rest the gyroscope reads ~0.5 dps, the input step is 2 dps
(1 dps with the other calibration): every idle window quantizes to zeros, and the output is exactly
0.25 / 0.25 / 0.25 / 0.25. Idle "wins" only because it is class 0 in a four-way tie. With the finer step, idle
becomes ±1 noise and is taken for `forward`. A fix needs a pre-processing change on both sides (scale or compress
the input, or int16 activations).

To deploy the model trained in the notebook instead, with the files of the TP's Colab code:

```sh
cd host
uv run python scripts/tp5_gesture.py --tflite model_int8.tflite --test-windows test_windows.h
```

## Run

```sh
cd firmware
cargo run --release -p gesture     # build, flash through the Arduino bootloader
```

Open the serial port (any baud rate: it is USB). On each connection the firmware prints:

1. **§3.1** the model information and `Arena used`;
2. **§3.2** one line per test window, to compare with the probabilities printed by `tp5_gesture.py`:
   `Window 1 | expected: side | predicted: side | probs: 0.000 0.996 0.000 0.000 | latency: … us`;
3. **§4** one line per window of 200 rows (~0.47 s): `<class> <prob> | <4 probs> | <latency> us`.

A Rev2 board (BMI270 instead of the LSM9DS1) runs parts 1 and 2 only and says so.

## Checked so far (no board yet)

- Builds (nanoglia 0.2.0): ~344 KB flash (80 KB of it is the model), ~26 KB static RAM (18 KB of it is the arena).
- The same `.tflite` and test windows run through nanoglia **on the PC** give exactly the TensorFlow Lite
  probabilities: 7 of the 8 operators on CMSIS-NN (`conv2d_s8`, `max_pool2d_s8`, `fully_connected_s8`,
  `softmax_s8`), the reshape in Rust; graph 2.3 KB + context 14.4 KB.

Reference probabilities (`idle side forward up_down`):

```
window 0 | expected: idle     | 0.250 0.250 0.250 0.250
window 1 | expected: side     | 0.000 0.996 0.000 0.000
window 2 | expected: forward  | 0.000 0.902 0.098 0.000
window 3 | expected: up_down  | 0.000 0.000 0.000 0.996
```

## On the board (to fill in)

- Flash / RAM, `Arena used`, latency per window: …
- Test windows: same probabilities as the reference? …
- (a) Stable predictions when the board does not move? During a gesture? …
- (b) Accuracy on the board vs. on the test set; most confused classes (expected: `forward` taken for `side`): …
- (c) Differences with the training data: different sensor than described (gyroscope, not accelerometer),
  row rate, orientation, amplitude, speed, gyroscope bias of each board: …
