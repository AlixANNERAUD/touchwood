# gesture: TP 5 with Rust and nanoglia

TP 5 deploys the TP 2 gesture classifier (idle, up_down, side, forward) on the Nano 33 BLE Sense.
The TP uses Arduino IDE + MicroTFLite; this crate does the same thing with Embassy and
[nanoglia](https://github.com/AlixANNERAUD/nanoglia), to test the whole Rust stack on a small model.

| TP (Arduino / MicroTFLite) | Here |
|---|---|
| `gesture_model.h` (C array, `alignas(16)`) | `models/gesture_model.tflite`, embedded with `include_bytes!` in a 16-byte aligned static |
| `test_windows.h` | `src/test_windows.rs` |
| Colab code of §2.1 and §2.2 | `host/scripts/tp5_gesture.py` (also prints the reference probabilities) |
| `tensorArena[kTensorArenaSize]` | `static mut MEMORY` given to a nanoglia `Arena` (4 KB) |
| `ModelInit` | `Graph::load_in` + `init_execution_context_in` |
| `ModelPrint*` | printed on connection: operators and kernels, input / output dims, type, scale, zero point, arena used |
| `ModelSetInput(x, i)` | `classify()`: quantizes the float window with the input scale and zero point (rounding like `np.round`) |
| `ModelRunInference` | `ctx.compute()` |
| `ModelGetOutput(j)` | `classify()`: dequantized probabilities |
| `Arduino_LSM9DS1` | `mod lsm9ds1`: same setup (119 Hz, ±4 g), `STATUS_REG` check, values in g |
| `micros()` sampling loop | `Ticker` every 10 ms (100 Hz); a tick without a new measurement adds no sample, as in the skeleton |

## Model

The repo has no TP 2 files, so `tp5_gesture.py` trains a **stand-in** on synthetic accelerations (board flat,
gravity on +z; up_down oscillates on z, side on x, forward on y), with the TP's defaults: `WINDOW = 50`, 3 axes,
interleaved order, classes in the TP's order. It is a 150 → 32 → 16 → 4 MLP (int8, 9.3 KB), 98.5 % int8
accuracy on synthetic test data. It shows that the chain works, **not** that it recognises real gestures.

To use the real TP 2 model, with the files produced by the TP's Colab code:

```sh
cd host
uv run python scripts/tp5_gesture.py --tflite model_int8.tflite --test-windows test_windows.h
```

Then check `WINDOW`, `CLASSES` and the flattening order at the top of `src/main.rs` against TP 2. The firmware
refuses a model whose input / output sizes do not match.

## Run

```sh
cd firmware
cargo run --release -p gesture     # build, flash through the Arduino bootloader
```

Open the serial port (any baud rate: it is USB). On each connection the firmware prints:

1. **§3.1** the model information and `Arena used` (to compare with Colab's `inp["shape"]`, `out["shape"]`);
2. **§3.2** one line per test window, to compare with the probabilities printed by `tp5_gesture.py`:
   `Window 0 | expected: idle | predicted: idle | probs: 0.992 0.004 0.004 0.000 | latency: … us`;
3. **§4** one line per 0.5 s window from the accelerometer: `<class> <prob> | <4 probs> | <latency> us`.

A Rev2 board (BMI270 instead of the LSM9DS1) runs parts 1 and 2 only and says so.

## Checked so far (no board yet)

- Builds: ~210 KB flash, ~8.3 KB static RAM (4 KB of it is the arena). nanoglia + CMSIS-NN make most of it;
  MicroTFLite would be in the same range (TFLite Micro).
- The same `.tflite` and test windows run through nanoglia **on the PC** give exactly the TensorFlow Lite reference
  probabilities (3 `fully_connected_s8` + `softmax_s8`, all CMSIS-NN; graph 1.4 KB + context 0.2 KB).

## On the board (to fill in)

- Flash / RAM reported by the build, `Arena used`, latency per window: …
- Test windows: same probabilities as the reference? …
- (a) Stable predictions when the board does not move? During a gesture? …
- (b) Accuracy on the board vs. on the test set; most confused classes: …
- (c) Differences with the training data (orientation, amplitude, speed, sensor calibration): …
