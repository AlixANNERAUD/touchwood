# firmware

Rust firmware for the Arduino Nano 33 BLE Sense (nRF52840), with no Arduino IDE or framework involved.

| Crate | Kind | What it does |
|---|---|---|
| `board` | lib | Board support: pins, sensor power-up, USB serial, bootloader entry, IMU detection |
| `acquisition` | bin | Streams the microphone continuously over USB serial, for dataset collection |
| `inference` | bin | Microphone window → int8 CNN ([nanoglia](https://github.com/AlixANNERAUD/nanoglia)) → zone id over USB serial |

**Status:** everything builds; **nothing has been tested on a board yet**.

Size at the latest build (release):

| Crate | Flash | Static RAM |
|---|---|---|
| `acquisition` | ~19 KB | ~8.7 KB |
| `inference` | ~183 KB | ~12 KB (8 KB of it is the nanoglia arena) |

Both are out of 960 KB of flash available to the app and 256 KB of RAM.

## Setup

```sh
rustup target add thumbv7em-none-eabihf      # also pinned in rust-toolchain.toml

# Arduino's fork of bossac (the mainline one cannot flash this board), no IDE needed:
curl -LO https://downloads.arduino.cc/tools/bossac-1.9.1-arduino2-linuxaarch64.tar.gz   # or ...-linux64.tar.gz on x86_64
tar xf bossac-*.tar.gz && sudo install bossac /usr/local/bin/   # or set BOSSAC=/path/to/bossac

# objcopy: any of rust-objcopy (cargo install cargo-binutils), llvm-objcopy, arm-none-eabi-objcopy
```

On Linux, your user needs access to `/dev/ttyACM*` (the `dialout` group on Debian).

## Build and flash

```sh
cargo build --release                       # all crates
cargo run --release -p acquisition          # build + flash (PORT=/dev/ttyACM1 to choose the port)
cargo run --release -p inference
```

`cargo run` calls [`scripts/flash.sh`](scripts/flash.sh), which:
1. converts the ELF to a raw `.bin`;
2. opens the port at 1200 baud: both the Arduino core and our firmware then reboot into the bootloader;
3. flashes with `bossac` at `0x10000` and resets.

If the board does not respond (for example after a crash), **double-tap reset** (the LED pulses) and run again.

## Board notes

The Arduino core silently does a few things at boot (`variants/ARDUINO_NANO33BLE/variant.cpp` in ArduinoCore-mbed). `board` does them for us:

| What | Pin | Note |
|---|---|---|
| Sensor rail (VDD_ENV) | P0.22 | **high drive**, otherwise the BMM150 browns out |
| Internal I²C pull-ups | P1.00 | ~10 ms after VDD_ENV |
| Internal I²C (IMU, ...) | SDA P0.14 / SCL P0.15 | 400 kHz |
| PDM microphone | PWR P0.17, CLK P0.26, DIN P0.25 | 1.28 MHz clock / 80 = 16 kHz |
| RGB LED (active low) | R P0.24, G P0.16, B P0.06 | |
| Bootloader entry | `GPREGRET = 0xB0` + reset | triggered by opening the port at 1200 baud |

- Memory layout ([`board/memory.x`](board/memory.x)): app at `0x10000`. The first 64 KB belong to the Arduino bootloader, **don't overwrite them**.
- IMU: the `LSM9DS1` (Rev1, I²C 0x6B) or the `BMI270` (Rev2, I²C 0x68) is detected from its ID register and reported in the info frame. BMI270 support will need its ~8 KB config blob from Bosch.
- There is no debug probe, so logs go over USB serial. With an SWD probe on the pads under the board (a Pi Pico running *debugprobe* works), `probe-rs` + `defmt` become possible. That needs a different `memory.x` and runner, and it would erase the bootloader if flashed at 0x0.

## acquisition: wire format

Little-endian frames, decoded by `host/src/touchwood/protocol.py`:

| Offset | Size | Field |
|---|---|---|
| 0 | 2 | magic `A5 5A` |
| 2 | 1 | type: `0` = info (ASCII), `1` = audio (`i16` PCM, 256 samples) |
| 3 | 1 | reserved |
| 4 | 4 | sequence number (`u32`, audio blocks numbered at capture time, so gaps mean drops) |
| 8 | 2 | payload length |
| 10 | n | payload |

At 16 kHz, 32 KB/s of audio goes over USB full speed easily. Capture never blocks: if the host falls behind, blocks are dropped and the gap shows up in the sequence numbers.

TODO: stream the IMU (after checking the board revision), and add a sync input pin for TDOA.

## inference: nanoglia

- [`inference/models/tap.tflite`](inference/models/tap.tflite) is a **placeholder** (random weights) exported by `host/scripts/export_tflite.py`.
- `host/notebooks/02_training.ipynb` exports the real pipeline as three models into `inference/models/`: `audio_encoder.tflite`, `imu_encoder.tflite` and `head.tflite`. They are not wired into `inference` yet.
- The model is embedded with `include_bytes!` and **loaded at run time** by nanoglia (`Graph::load_in`): the weights are read in place from flash, the graph and activations live in a static 8 KB `Arena` (the placeholder needs ~4.6 KB; grow it for the real models) (no heap). Re-exporting the model rebuilds the firmware.
- Because loading happens at run time, a model could later come over BLE or serial instead of being built in.
- On connection, the firmware prints `nanoglia: arena <used>/<capacity> B` (use it to shrink the arena), or `model load failed: <reason>` if the model is rejected.
- Flash grows from ~34 KB (MicroFlow) to ~183 KB: the price of a general runtime plus the CMSIS-NN kernels.
- The op-set constraints are listed in the root README (§7). The export script checks them.
- Output: one line per window, `zone=<z> conf=<p>% t=<latency>us`.

TODO: tap onset detection on-device (the same energy trigger as `host/src/touchwood/taps.py`), the real front-end normalisation, and the IMU branch and fusion head.
