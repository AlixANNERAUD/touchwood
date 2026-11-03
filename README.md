# touchwood

> Turn any table into a touch surface: two Arduino Nano 33 BLE Sense boards stuck under a table recognise **where** you tapped, using compressed neural networks running on the microcontrollers and talking over BLE.

**Status:** internal working notes, early stage. Nothing here is final.

Course project for *IA embarquée* (Embedded AI), 2026–2027. Team of 2, about one month.

---

## 1. Idea

No triangulation is required. Every location on a surface leaves its own vibro-acoustic **fingerprint**, and a small CNN learns to recognise it. A **second board** at the other end of the table adds spatial cues, such as the energy ratio between the boards (see [§3.1](#31-two-board-setup)). We frame the problem as **classification over zones**, starting with a 2×2 grid and moving to 3×3. Regression of (x, y) is a stretch goal.

The core of the project is to **fit the model onto the microcontroller**, applying as many techniques from the course as possible and measuring the effect of each one.

### Why a single sensor can work (hypotheses to validate)

**Microphone** (board glued under the surface, so it picks up structure-borne sound):

- **Attenuation**: high frequencies die out faster with distance, so the spectral tilt depends on distance.
- **Dispersion**: flexural waves in a plate are dispersive (high frequencies travel faster), so the waveform smears out with distance.
- **Edge reflections**: the delays of the echoes off the table edges depend on where the tap happened.
- **Modal excitation**: a tap on a node of a mode does not excite that mode, so each position excites a different mix of resonances.
- **Structure vs. air arrival**: the same tap arrives through wood (~3000 m/s) and through air (343 m/s). At 50 cm the two are ~1.3 ms apart (~21 samples at 16 kHz), which could act as a distance cue. *Unverified.*

**IMU**:

- When the flexural wave passes under the board, the surface tilts locally. The gyroscope measures this, and the ratio of rotation on the X and Y axes may encode the **direction** of arrival.
- ⚠️ This is the weakest hypothesis. The IMU sample rate is at most ~1 kHz and a heavy table barely moves. A lighter, thinner surface (desk panel, shelf, door) should help a lot.

Working assumption: **the microphone carries most of the information and the IMU is a bonus.** A per-modality ablation will tell.

---

## 2. Hardware

| Device | Role |
|---|---|
| 2× Arduino Nano 33 BLE **Sense** (nRF52840, Cortex-M4F @ 64 MHz, **1 MB flash / 256 KB RAM**) ✅ confirmed | Target: always-on sensing and inference. One board is the *main* node, the other is a *secondary* node. |
| Other Arduino Nano boards (exact model **TBD**) | If it is a Nano ESP32, it could be used for Wi-Fi CSI (see [§11](#11-parked-ideas)) |
| Raspberry Pi | Host for the demo (receives zone events), cross-platform benchmark |
| Jetson Nano | Teacher training / inference, TensorRT benchmark |
| *(optional)* piezo disc(s) on an analog pin | Plan B contact microphone; two of them would allow real time-difference-of-arrival (TDOA) |

**TODO:** check the board revision.
- **Rev1**: LSM9DS1 IMU. The Arduino library defaults to ~119 Hz; it can go to 952 Hz by setting registers.
- **Rev2**: BMI270 + BMM150. The accelerometer goes up to ~1.6 kHz and the gyroscope higher.

---

## 3. Pipeline

Single-node pipeline, which runs on each board:

```
Mic (PDM, 16 kHz) ──► energy threshold (always-on, near-zero cost)
                          │ tap detected
                          ▼
       capture window: ~50 ms audio + ~100 ms IMU (aligned on onset)
                          │
           ┌──────────────┴──────────────┐
     Audio branch                   IMU branch
     (Conv1D on raw / log-mel)      (Conv1D on 6 axes)
           └──────────────┬──────────────┘
                 fusion → Dense → zone id / embedding
```

The board sleeps between taps, which gives us an energy-per-inference argument.

### 3.1 Two-board setup

```
   ┌──────────────── table (top view) ────────────────┐
   │ [B] secondary                         [A] main   │
   │                ·  ·  ·                           │
   │                ·  ·  ·   tap zones               │
   │                ·  ·  ·                           │
   └──────────────────────────────────────────────────┘
   B ──BLE (raw / embedding / logits)──► A ──serial / BLE──► Raspberry Pi
   (optional wire A↔B: GPIO sync pulse for µs-level alignment)
```

Put the boards at **opposite ends of the long axis** (or diagonally): the further apart they are, the stronger the cues.

**What the second board adds, from easiest to hardest:**

1. **Inter-board energy ratio** ⭐. The closer board gets the stronger signal. The broadband and per-band energy ratios tell which side the tap was on and roughly how far.
   - Only **coarse sync** is needed: we just have to match the same tap on both boards, within a few ms. BLE is enough.
   - The cue **transfers across tables** much better than raw fingerprints, which helps the generalisation goal.
2. **Two fingerprints instead of one.** Each board sees the tap from a different viewpoint, so they disambiguate each other.
3. **Time difference of arrival (TDOA).** Flexural waves in wood travel at roughly a few hundred m/s and are dispersive. A 10 cm path difference is ~200 µs, or ~3 samples at 16 kHz.
   - TDOA needs clocks aligned to **tens of µs**. ArduinoBLE jitter is around the **ms**, so it is unusable for this.
   - With only 2 sensors, TDOA gives a hyperbola, not a point. It is still useful combined with the fingerprints.
   - Options to get the sync:
     - ✅ **A wire between the boards** (GPIO sync pulse). Trivial and near-perfect.
     - 🧪 **nRF52840 radio timestamping**: wire the RADIO address event to a TIMER capture through PPI, which allows sub-µs sync over the air. This means leaving the Arduino BLE stack (Zephyr / Embassy / bare radio). It is a good fit for the Rust track.
     - 💡 **A sync-free cue**: if the structure-vs-air arrival gap is confirmed (§1), each board measures its *own* distance to the tap. Two distances intersect as two circles, which gives a position without any inter-board sync.

**Fusion strategies (distributed inference / split computing):** what does B send to A?

| B sends | Payload | Trade-off |
|---|---|---|
| Raw window (50 ms @ 16 kHz, int16) | ~1.6 KB | Early fusion. Best accuracy potential, highest radio energy and latency. |
| int8 embedding from B's feature extractor | ~32 B | **Split computing**: each node runs the extractor and A runs the fusion head. |
| Per-zone probabilities (int8) | ~9 B | Late fusion. Cheapest, least expressive. |

To measure for each strategy: accuracy, bytes on air, radio energy, end-to-end latency (tap → decision on A), and flash / RAM on each node.

A third board would make 2D TDOA possible. That is a stretch goal, not the baseline.

### Generalising to a new table (stretch goal)

- **Baseline:** one model per surface, retrained for each surface.
- **Stretch:** train an **embedding extractor** (prototypical network / metric learning) on data from 5–10 different surfaces. On a new table, the user taps each zone 5–10 times and the board stores the mean embedding per zone. Inference is then a nearest-prototype lookup.
- This means **no backprop on the MCU**, and calibration takes about 30 s.

---

## 4. Course techniques we apply

The deliverable is a **cumulative ablation table**: the same model, step by step, measured **on the device**.

| Step | Accuracy | Params | Flash | RAM (arena) | On-device latency | Energy / inference |
|---|---|---|---|---|---|---|
| Teacher (float32, PC/Jetson) | | | ❌ doesn't fit | | | |
| Student, trained from scratch | | | | | | |
| Student + knowledge distillation | | | | | | |
| + structured pruning / low-rank (SVD) | | | | | | |
| + weight clustering | | | | | | |
| + QAT int8 | | | ✅ | | | |

Checklist:

- [ ] Efficient architecture (1D CNN, depthwise-separable convolutions)
- [ ] Knowledge distillation (big teacher → small student; temperature, α sweep)
- [ ] Pruning: unstructured (magnitude) **vs.** structured (channels). Show that only structured pruning reduces latency and RAM under TFLite Micro.
- [ ] Low-rank factorisation (SVD) of dense / convolution layers
- [ ] Weight sharing / clustering, plus compression (zip / Huffman)
- [ ] Activation sparsity (ReLU threshold, L1 penalty on activations)
- [ ] Quantization: PTQ (dynamic range, float16, full int8) **vs.** QAT
- [ ] Deployment: TFLite Micro (C++), and MicroFlow (Rust) if time allows
- [ ] Metrics: accuracy / confusion matrix, size, peak RAM, latency, energy
- [ ] Pareto plot: accuracy vs. size, and accuracy vs. latency
- [ ] Classical baseline: Location Template Matching (cross-correlation + k-NN) vs. CNN, comparing both accuracy and cost
- [ ] Cross-platform benchmark of the same model: Arduino vs. Raspberry Pi vs. Jetson
- [ ] Distributed inference over BLE: early vs. split (embedding) vs. late fusion (§3.1)
- [ ] Ablations by sensor (board A only / board B only / A+B) and by modality (mic / IMU)

---

## 5. Roadmap

### Week 1: feasibility (go / no-go)

- [ ] Identify the board revision (Rev1 / Rev2) of both Sense boards; get mic and IMU streaming over serial at the highest usable rates.
- [ ] Data logger: firmware that streams raw samples, plus a Python script that saves labelled taps from **both boards at once** (both on USB).
  - PC-side alignment is fine for energy ratios and fingerprints, but **not** for TDOA, because of USB jitter.
  - If we want to test TDOA, add the GPIO sync wire from day one.
- [ ] Record **4 corners × 30 taps** on one surface, with the boards at opposite ends.
- [ ] Plot spectrograms and gyroscope traces per zone. Can we see a difference by eye?
- [ ] Quick k-NN / random forest on:
  - **board A only**, **board B only** and **A+B** (energy ratio plus both feature sets);
  - **mic only**, **IMU only** and **both**.
- **Go** if above ~90 % on 4 zones. **Adopt the two-board setup** if A+B clearly beats a single board. **Otherwise**, try a lighter surface or a piezo, or pivot.
- [ ] Minimal BLE link (ArduinoBLE): B notifies A with a small payload on each tap. Measure the latency.
- [ ] TFLite Micro "hello world" running on the board.

### Week 2: data and first models

- [ ] Onset detection and windowing (on PC first, then ported to firmware).
- [ ] Dataset v1: 3×3 grid, varied tap force, finger vs. knuckle, several sessions or days, with and without objects on the table.
- [ ] Teacher model (large, PC/Jetson) and baseline student.
- [ ] Deployment pipeline and on-device measurement harness (latency, arena size, energy).

### Week 3: compression

- [ ] Distillation, then pruning, SVD, clustering and QAT, filling the ablation table.
- [ ] Per-modality ablation (mic / IMU / both) and per-board ablation (A / B / A+B).
- [ ] Fusion strategies over BLE (raw / embedding / logits): accuracy vs. bytes vs. energy vs. latency.
- [ ] *(stretch)* Multi-surface dataset and embedding plus on-device calibration.

### Week 4: integration and report

- [ ] Real-time demo: tap zones control something on the Raspberry Pi (music player, game, slides).
- [ ] Robustness study (moved objects, a different person tapping).
- [ ] Report and presentation.

### Suggested split

- **A:** data, models, distillation and compression (Python).
- **B:** firmware, deployment and on-device measurements (C++ / Rust).

Swap and review each other's work regularly.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| IMU signal too weak for direction | Lighter surface, knuckle taps, piezo plan B, accept a mic-only model |
| Fingerprint drift (objects on table, board moved) | Varied data, augmentation, few-shot calibration (stretch) |
| Mic / IMU synchronisation (different rates, I²C latency) | Align both windows on the detected onset; use the IMU FIFO |
| Inter-board sync too poor for TDOA (BLE jitter ~ms) | Rely on energy ratio and fingerprints, which only need coarse sync; GPIO sync wire; radio timestamping as a stretch goal |
| BLE link adds latency or drops packets | Late fusion fallback (A decides alone if B's message is late); measure the end-to-end latency |
| Model already tiny, so little left to compress | Start from a deliberately large teacher; report the full trade-off curve |
| Rust toolchain eats the schedule | C++ / TFLite Micro is the reference path; Rust is a parallel track, not a blocker |

---

## 7. Rust track (optional, parallel)

- **HAL:** Embassy (`embassy-nrf`), which has a PDM driver and async I²C.
- **Inference:** [MicroFlow](https://arxiv.org/abs/2409.19432) (`.tflite` → Rust at compile time, tested on nRF52840). Its operator set is limited: check that Conv1D, which TFLite lowers to Conv2D, and the pooling layers we use are supported. A hand-written int8 engine is a fallback with real teaching value.
- **Flashing:** double-tap reset, then Arduino's fork of `bossac` (the mainline one does not work), and link after the bootloader (check the offset). Alternatively, use an SWD probe (Pi Pico + *debugprobe*) with `probe-rs`.
- **Gotcha:** on the Sense board, sensor power and the I²C pull-ups are switched by GPIOs that the Arduino core enables silently. We must do it ourselves; check the schematic or the Arduino variant file.

---

## 8. Python environment

Managed with [uv](https://docs.astral.sh/uv/), on **Python 3.13**. Python 3.14+ is not an option yet: TensorFlow, ExecuTorch and the PyTorch→TFLite converter (LiteRT Torch) have no wheels for it, and LiteRT Torch has no Linux aarch64 wheels at all.

PyTorch is an extra, so pick exactly one:

```sh
uv sync --extra cpu         # CPU-only PyTorch (Linux / Windows, incl. aarch64)
uv sync --extra cuda        # NVIDIA GPU, CUDA 12.6 wheels (x86_64)
uv sync --extra mps         # Apple Silicon, native macOS only
uv run jupyter lab          # or select the .venv kernel in VS Code
uv run ruff check .         # lint
```

`tensorflow-model-optimization` (pruning / clustering) needs legacy Keras. Set this **before** importing TensorFlow:

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
```

---

## 9. Proposed repo layout

```
firmware/
  logger/        # raw mic + IMU streaming for data collection
  inference/     # TFLite Micro deployment (C++)
  rust/          # optional Embassy + MicroFlow port
data/            # recordings (large files git-ignored / LFS)
notebooks/       # exploration, feasibility study
training/        # teacher, student, distillation, compression scripts
models/          # exported .tflite / .h / ONNX
bench/           # on-device measurement scripts and results
demo/            # Raspberry Pi side
report/
```

---

## 10. References

- Knock&Tap: classification and localisation of knock and tap gestures (4-mic array): <https://pure.seoultech.ac.kr/en/publications/knockamptap-classification-and-localization-of-knock-and-tap-gest/>
- MIT, Passive acoustic sensing for tracking knocks on large displays: <https://aspirin.media.mit.edu/responsive/wp-content/uploads/sites/4/2013/12/Passive-Acoustic-Sensing-for-Tracking-Knocks-Atop-Large-Interactive-Displays.pdf>
- TapLock, tap detection with an accelerometer on a Nano 33 BLE Sense: <https://www.hackster.io/taplock/taplock-a-bike-lock-with-machine-learning-85641c>
- MicroFlow, a Rust TinyML inference engine: <https://arxiv.org/abs/2409.19432>
- Burn, `no_std` mode: <https://burn.dev/books/burn/advanced/no-std.html>
- Zephyr board notes for the Nano 33 BLE (flashing, bossac): <https://developer.nordicsemi.com/nRF_Connect_SDK/doc-legacy/1.6.1/zephyr/boards/arm/arduino_nano_33_ble/doc/index.html>
- Course material: TinyML, quantization, pruning / clustering (TP3), knowledge distillation (TP4)
- WiFinger, fine-grained finger gesture recognition with Wi-Fi: <https://arxiv.org/abs/2106.00857>
- Espressif TFLite Micro port with ESP-NN kernels: <https://github.com/espressif/esp-tflite-micro>

---

## 11. Parked ideas

Not in the plan, kept here for reference.

- **BLE "radar" with the Nano's radio.** Not viable.
  - Range resolution is c / 2B: ~75 m for one 2 MHz BLE channel, ~1.9 m across the whole 80 MHz band.
  - The nRF52840 only reports RSSI. It supports neither AoA/AoD direction finding nor BLE 6.0 Channel Sounding (those are on the nRF52833 / nRF5340 and nRF54L series).
  - A multi-channel RSSI fingerprint of the *arm* position might separate left from right. That would be a possible negative-result experiment at most.
- **Wi-Fi CSI with ESP32 boards** (Nano ESP32 / ESP32-S3).
  - CSI gives ~52 subcarriers (amplitude + phase) per packet, which is far richer than RSSI. It senses the *hand / arm position*, not the tap itself, so it complements the vibration sensing (vibrations tell *when* the tap happened, CSI tells *where* the hand is).
  - Place TX and RX on opposite sides of the table so the hand crosses the first Fresnel zone (radius ~18 cm at mid-link for 1 m at 2.4 GHz).
  - Downsides: very sensitive to the environment (people moving, posture), and it doubles the acquisition chain.
  - If we try it, run the same week-1 test on CSI only.
  - An alternative without new hardware is Nexmon CSI on a Raspberry Pi 3B+ / 4.
  - Prior art: **RuView / "WiFi DensePose"**, an open-source project that went viral. It claims to estimate body pose, breathing and heart rate, even through walls, from a router plus several ESP32-S3 nodes reading CSI ([coverage](https://www.opensourceforu.com/?p=97115), [PyPI](https://pypi.org/project/wifi-densepose/2.0.0a1/), [a GitHub fork](https://github.com/meterup/ruview)).
    - Its figures are self-reported and unverified (some, like "54K fps", look doubtful), and it is beta software.
    - The underlying science is solid: it builds on CMU's *DensePose from WiFi* (Geng et al.).
    - It confirms that ESP32-S3 + CSI is the right stack, but these systems sense whole-body motion with several nodes. A finger tap on a ~20 cm zone is much finer-grained.
- **Magnetic stylus + magnetometer**: a real localisation technique, but short range (~10–20 cm) and it changes the concept.
- **Capacitive electrodes** (copper tape under the table): proximity sensing at a few cm range; it needs several electrodes to localise.
