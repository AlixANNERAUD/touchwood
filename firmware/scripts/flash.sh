#!/usr/bin/env bash
# Cargo runner: `cargo run --release -p <crate>` lands here with the ELF path.
# Converts it to a raw binary and flashes it through the Arduino bootloader with bossac.
#
# Env vars:
#   PORT     serial port of the board        (default: /dev/ttyACM0)
#   BOSSAC   Arduino fork of bossac           (default: bossac on PATH)
#   OBJCOPY  objcopy that understands ARM ELF (default: rust-objcopy, llvm-objcopy or arm-none-eabi-objcopy)
set -euo pipefail

elf="$1"
bin="${elf}.bin"
port="${PORT:-/dev/ttyACM0}"
bossac="${BOSSAC:-bossac}"

if [[ -z "${OBJCOPY:-}" ]]; then
  for candidate in rust-objcopy llvm-objcopy arm-none-eabi-objcopy; do
    if command -v "$candidate" >/dev/null; then OBJCOPY="$candidate"; break; fi
  done
fi
"${OBJCOPY:?no objcopy found, install cargo-binutils or binutils-arm-none-eabi}" -O binary "$elf" "$bin"
echo "image: $bin ($(stat -c %s "$bin") bytes)"

# "1200 baud touch": both the Arduino core and our firmware reboot into the bootloader.
# If the board is stuck, double-tap reset instead.
if [[ -e "$port" ]]; then
  stty -F "$port" 1200 || true
  sleep 1.5
fi
for _ in $(seq 1 20); do [[ -e "$port" ]] && break; sleep 0.25; done

"$bossac" -d --port="$(basename "$port")" -U -i -e -w "$bin" -R
