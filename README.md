# touchwood

> Turn any table into a touch surface: an Arduino Nano 33 BLE Sense stuck under a table recognises **where** you tapped, using a compressed neural network running on the microcontroller.

**Status:** internal working notes, early stage. Nothing here is final.

Course project for *IA embarquée* (Embedded AI), 2026–2027. Team of 2, about one month.

---

## 1. Idea

One board, one microphone, one IMU, no triangulation. Every location on a surface leaves its own vibro-acoustic **fingerprint**, and a small CNN learns to recognise it. We frame the problem as **classification over zones**, starting with a 2×2 grid and moving to 3×3. Regression of (x, y) is a stretch goal.

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
| Arduino Nano 33 BLE **Sense** (nRF52840, Cortex-M4F @ 64 MHz, **1 MB flash / 256 KB RAM**) | Target: always-on sensing and inference |
| Raspberry Pi | Host for the demo (receives zone events), cross-platform benchmark |
| Jetson Nano | Teacher training / inference, TensorRT benchmark |
| *(optional)* piezo disc(s) on an analog pin | Plan B contact microphone; two of them would allow real time-difference-of-arrival (TDOA) |

**TODO:** check the board revision.
- **Rev1**: LSM9DS1 IMU. The Arduino library defaults to ~119 Hz; it can go to 952 Hz by setting registers.
- **Rev2**: BMI270 + BMM150. The accelerometer goes up to ~1.6 kHz and the gyroscope higher.

---

## 3. Pipeline

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
                 fusion → Dense → zone id
                          │
                          ▼
              serial / BLE → Raspberry Pi (demo app)
```

The board sleeps between taps, which gives us an energy-per-inference argument.

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

---

## 5. Roadmap

### Week 1: feasibility (go / no-go)

- [ ] Identify the board revision; get mic and IMU streaming over serial at the highest usable rates.
- [ ] Data logger: firmware that streams raw samples, plus a Python script that saves labelled taps.
- [ ] Record **4 corners × 30 taps** on one surface.
- [ ] Plot spectrograms and gyroscope traces per zone. Can we see a difference by eye?
- [ ] Quick k-NN / random forest on **mic only**, **IMU only** and **both**.
- **Go** if above ~90 % on 4 zones. **Otherwise**, try a lighter surface or a piezo, or pivot.
- [ ] TFLite Micro "hello world" running on the board.

### Week 2: data and first models

- [ ] Onset detection and windowing (on PC first, then ported to firmware).
- [ ] Dataset v1: 3×3 grid, varied tap force, finger vs. knuckle, several sessions or days, with and without objects on the table.
- [ ] Teacher model (large, PC/Jetson) and baseline student.
- [ ] Deployment pipeline and on-device measurement harness (latency, arena size, energy).

### Week 3: compression

- [ ] Distillation, then pruning, SVD, clustering and QAT, filling the ablation table.
- [ ] Per-modality ablation (mic / IMU / both).
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
| Model already tiny, so little left to compress | Start from a deliberately large teacher; report the full trade-off curve |
| Rust toolchain eats the schedule | C++ / TFLite Micro is the reference path; Rust is a parallel track, not a blocker |

---

## 7. Rust track (optional, parallel)

- **HAL:** Embassy (`embassy-nrf`), which has a PDM driver and async I²C.
- **Inference:** [MicroFlow](https://arxiv.org/abs/2409.19432) (`.tflite` → Rust at compile time, tested on nRF52840). Its operator set is limited: check that Conv1D, which TFLite lowers to Conv2D, and the pooling layers we use are supported. A hand-written int8 engine is a fallback with real teaching value.
- **Flashing:** double-tap reset, then Arduino's fork of `bossac` (the mainline one does not work), and link after the bootloader (check the offset). Alternatively, use an SWD probe (Pi Pico + *debugprobe*) with `probe-rs`.
- **Gotcha:** on the Sense board, sensor power and the I²C pull-ups are switched by GPIOs that the Arduino core enables silently. We must do it ourselves; check the schematic or the Arduino variant file.

---

## 8. Proposed repo layout

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

## 9. References

- Knock&Tap: classification and localisation of knock and tap gestures (4-mic array): <https://pure.seoultech.ac.kr/en/publications/knockamptap-classification-and-localization-of-knock-and-tap-gest/>
- MIT, Passive acoustic sensing for tracking knocks on large displays: <https://aspirin.media.mit.edu/responsive/wp-content/uploads/sites/4/2013/12/Passive-Acoustic-Sensing-for-Tracking-Knocks-Atop-Large-Interactive-Displays.pdf>
- TapLock, tap detection with an accelerometer on a Nano 33 BLE Sense: <https://www.hackster.io/taplock/taplock-a-bike-lock-with-machine-learning-85641c>
- MicroFlow, a Rust TinyML inference engine: <https://arxiv.org/abs/2409.19432>
- Burn, `no_std` mode: <https://burn.dev/books/burn/advanced/no-std.html>
- Zephyr board notes for the Nano 33 BLE (flashing, bossac): <https://developer.nordicsemi.com/nRF_Connect_SDK/doc-legacy/1.6.1/zephyr/boards/arm/arduino_nano_33_ble/doc/index.html>
- Course material: TinyML, quantization, pruning / clustering (TP3), knowledge distillation (TP4)
