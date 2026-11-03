"""Guided dataset recording from one or two boards running the `acquisition` firmware.

Full procedure: host/DATASET.md. Typical session with two boards:

    uv run python scripts/record.py --session 20261110_desk1_s1 \\
        --ports A=/dev/ttyACM0,B=/dev/ttyACM1 --zones 9 --taps 25 --condition finger

For each zone, in a random order, it asks you to tap, records all boards at once, counts the
taps, and checks the take (dropped blocks, clipping, A/B agreement). You can keep or redo the
take. Single take, no prompt:

    uv run python scripts/record.py --session 20261110_desk1_s1 --label noise_speech --seconds 60

Output, one file per board and take, in data/raw/<session>/:
    <board>_<label>_<stamp>.npz   audio (int16), seq, fs, info, t0 (host time of the first block)
    takes.csv                     one line per kept take (the session log)
    session.json                  setup metadata, written on the first run (see --surface, ...)
"""

import argparse
import csv
import json
import random
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial

from touchwood.protocol import TYPE_AUDIO, TYPE_INFO, Decoder
from touchwood.taps import detect_onsets, pair_onsets

FS = 16_000
DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


class BoardReader(threading.Thread):
    """Reads one serial port in the background and keeps the audio blocks of the current take."""

    def __init__(self, name: str, port: str):
        super().__init__(daemon=True)
        self.name, self.port = name, port
        self.serial = serial.Serial(port, baudrate=115200, timeout=0.05)  # baud is ignored by USB CDC
        self.decoder = Decoder()
        self.info = ""
        self.lock = threading.Lock()
        self.recording = False
        self.blocks, self.seqs, self.t0 = [], [], None
        self.alive = True

    def run(self) -> None:
        while self.alive:
            for frame in self.decoder.feed(self.serial.read(4096)):
                if frame.kind == TYPE_INFO:
                    self.info = frame.text()
                elif frame.kind == TYPE_AUDIO:
                    with self.lock:
                        if self.recording:
                            if self.t0 is None:
                                self.t0 = time.time()
                            self.blocks.append(frame.audio())
                            self.seqs.append(frame.seq)

    def start_take(self) -> None:
        with self.lock:
            self.blocks, self.seqs, self.t0 = [], [], None
            self.recording = True

    def stop_take(self) -> tuple[np.ndarray, np.ndarray, float | None]:
        with self.lock:
            self.recording = False
            audio = np.concatenate(self.blocks) if self.blocks else np.empty(0, np.int16)
            return audio, np.asarray(self.seqs, dtype=np.uint32), self.t0

    def live_taps(self) -> int:
        with self.lock:
            audio = np.concatenate(self.blocks) if self.blocks else np.empty(0, np.int16)
        return len(detect_onsets(audio, FS)) if len(audio) > FS // 2 else 0

    def close(self) -> None:
        self.alive = False
        self.join(timeout=1)
        self.serial.close()


def parse_ports(spec: str) -> dict[str, str]:
    """'A=/dev/ttyACM0,B=/dev/ttyACM1' -> {'A': ..., 'B': ...}."""
    return dict(item.split("=", 1) for item in spec.split(","))


def check_take(takes: dict[str, tuple], target: int | None) -> tuple[dict, list[str]]:
    """Quality report of one take. Returns (stats for takes.csv, list of warnings)."""
    stats, warnings, onsets = {}, [], {}
    for b, (audio, seq, _) in takes.items():
        dropped = int(np.sum(np.diff(seq.astype(np.int64)) - 1)) if len(seq) > 1 else 0
        clipped = float(np.mean(np.abs(audio.astype(np.int32)) >= 32000)) if len(audio) else 0.0
        onsets[b] = detect_onsets(audio, FS)
        stats[f"taps_{b}"], stats[f"dropped_{b}"] = len(onsets[b]), dropped
        if len(audio) == 0:
            warnings.append(f"board {b}: no audio received (firmware? cable?)")
        if dropped:
            warnings.append(f"board {b}: {dropped} dropped blocks (USB too slow, close other apps)")
        if clipped > 1e-4:
            warnings.append(f"board {b}: clipping on {clipped:.2%} of samples (tap softer)")
        if target and abs(len(onsets[b]) - target) > max(2, target // 10):
            warnings.append(f"board {b}: {len(onsets[b])} taps detected, {target} expected (double taps? too soft?)")
    if len(onsets) == 2:
        a, b = onsets.values()
        stats["paired"] = len(pair_onsets(a, b, FS))
        if stats["paired"] < 0.9 * min(len(a), len(b)):
            warnings.append(f"only {stats['paired']} taps paired between boards (one board not hearing the taps?)")
    return stats, warnings


def save_take(session_dir: Path, label: str, takes: dict, readers: dict, stats: dict, args) -> str:
    stamp = datetime.now().strftime("%H%M%S")
    for b, (audio, seq, t0) in takes.items():
        np.savez_compressed(
            session_dir / f"{b}_{label}_{stamp}.npz",
            audio=audio,
            seq=seq,
            fs=FS,
            info=readers[b].info,
            t0=np.nan if t0 is None else t0,
        )
    log = session_dir / "takes.csv"
    row = {
        "stamp": stamp,
        "label": label,
        "condition": args.condition,
        "operator": args.operator,
        "boards": "+".join(takes),
        "seconds": round(max(len(t[0]) for t in takes.values()) / FS, 1),
        **stats,
        "notes": args.notes,
    }
    new = not log.exists()
    with log.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
    return stamp


def record_one(label: str, readers: dict, args, prompt: bool) -> dict:
    """Record one take on all boards. Stops after --seconds, or once --taps taps are heard."""
    if prompt:
        input(f"\n▶ {label}: press Enter, then tap {args.taps} times (≈1 per second, irregular)... ")
    for r in readers.values():
        r.start_take()
    t_end = time.monotonic() + args.seconds
    first = next(iter(readers.values()))
    while time.monotonic() < t_end:
        time.sleep(0.25)
        n = first.live_taps()
        print(f"\r  recording {label}: {n} taps, {t_end - time.monotonic():4.1f} s left ", end="", flush=True)
        if prompt and args.taps and n >= args.taps:
            time.sleep(1.0)  # let the last tap ring out
            break
    print()
    return {b: r.stop_take() for b, r in readers.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="<date>_<surface>_<setup>, e.g. 20261110_desk1_s1")
    ap.add_argument("--ports", default="A=/dev/ttyACM0", help="board=port pairs, e.g. A=/dev/ttyACM0,B=/dev/ttyACM1")
    ap.add_argument("--zones", type=int, help="guided mode: record zones z0..z<N-1> in a random order")
    ap.add_argument("--label", help="single take with this label (z3, noise_speech, ...)")
    ap.add_argument("--taps", type=int, default=25, help="taps per take (guided mode stops when reached)")
    ap.add_argument("--seconds", type=float, default=45.0, help="maximum duration of a take")
    ap.add_argument("--condition", default="finger", help="tap condition: finger, knuckle, soft, objects, ...")
    ap.add_argument("--operator", default="", help="who is tapping")
    ap.add_argument("--notes", default="")
    ap.add_argument("--surface", default="", help="session.json: surface description (first run)")
    ap.add_argument("--layout", default="", help="session.json: table size, grid, board positions (first run)")
    args = ap.parse_args()
    if bool(args.zones) == bool(args.label):
        ap.error("give either --zones (guided) or --label (single take)")

    session_dir = DATA / args.session
    session_dir.mkdir(parents=True, exist_ok=True)
    meta = session_dir / "session.json"
    if not meta.exists():
        info = {
            "session": args.session,
            "created": datetime.now().isoformat(timespec="seconds"),
            "surface": args.surface,
            "layout": args.layout,
            "ports": args.ports,
        }
        meta.write_text(json.dumps(info, indent=2) + "\n")

    readers = {b: BoardReader(b, p) for b, p in parse_ports(args.ports).items()}
    for r in readers.values():
        r.start()
    time.sleep(1.0)
    for b, r in readers.items():
        print(f"board {b} on {r.port}: {r.info or 'no info frame yet (reopen the port if this persists)'}")

    labels = [args.label] if args.label else random.sample([f"z{i}" for i in range(args.zones)], args.zones)
    if args.zones:
        print("zone order for this session:", " ".join(labels))
    try:
        for label in labels:
            while True:
                takes = record_one(label, readers, args, prompt=bool(args.zones))
                stats, warnings = check_take(takes, args.taps if args.zones else None)
                print("  " + ", ".join(f"{k}={v}" for k, v in stats.items()))
                for w in warnings:
                    print("  ⚠", w)
                answer = input("  [k]eep / [r]edo / [s]kip? ").strip().lower() if args.zones else "k"
                if answer.startswith("r"):
                    continue
                if not answer.startswith("s"):
                    stamp = save_take(session_dir, label, takes, readers, stats, args)
                    print(f"  saved {label}_{stamp} -> {session_dir}")
                break
    finally:
        for r in readers.values():
            r.close()


if __name__ == "__main__":
    main()
