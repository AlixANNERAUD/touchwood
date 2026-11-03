"""Record a labelled session from the `acquisition` firmware.

Tap repeatedly on ONE zone while it records, then run it again for the next zone.

    uv run python scripts/record.py --label z0 --seconds 30
    uv run python scripts/record.py --label z0 --seconds 30 --board B --port /dev/ttyACM1

Output: data/raw/<session>/<board>_<label>_<timestamp>.npz with
    audio (int16), seq (block sequence numbers), fs, info (firmware banner).
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial

from touchwood.protocol import TYPE_AUDIO, TYPE_INFO, Decoder

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="zone label, e.g. z0..z8, or 'noise'")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--board", default="A", help="board id when recording with two boards")
    ap.add_argument("--session", default=datetime.now().strftime("%Y%m%d"))
    args = ap.parse_args()

    decoder, blocks, seqs, info = Decoder(), [], [], ""
    with serial.Serial(args.port, baudrate=115200, timeout=0.1) as port:  # baud is ignored by USB CDC
        port.reset_input_buffer()
        t_end = time.monotonic() + args.seconds
        print(f"recording '{args.label}' for {args.seconds:.0f} s, tap now...")
        while time.monotonic() < t_end:
            for frame in decoder.feed(port.read(4096)):
                if frame.kind == TYPE_INFO:
                    info = frame.text()
                    print("firmware:", info)
                elif frame.kind == TYPE_AUDIO:
                    blocks.append(frame.audio())
                    seqs.append(frame.seq)

    if not blocks:
        raise SystemExit("no audio received: is the acquisition firmware flashed?")
    seq = np.asarray(seqs, dtype=np.uint32)
    gaps = int(np.sum(np.diff(seq.astype(np.int64)) - 1)) if len(seq) > 1 else 0
    out = DATA / args.session / f"{args.board}_{args.label}_{datetime.now():%H%M%S}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, audio=np.concatenate(blocks), seq=seq, fs=16_000, info=info)
    print(f"saved {sum(map(len, blocks)) / 16_000:.1f} s of audio ({gaps} dropped blocks) -> {out}")


if __name__ == "__main__":
    main()
