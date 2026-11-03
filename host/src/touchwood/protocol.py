"""Decoder for the frames sent by the `acquisition` firmware (see firmware/acquisition/src/main.rs).

Frame (little-endian): magic A5 5A | type u8 | reserved u8 | seq u32 | len u16 | payload.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"\xa5\x5a"
HEADER = struct.Struct("<2sBBIH")
TYPE_INFO = 0
TYPE_AUDIO = 1


@dataclass
class Frame:
    kind: int
    seq: int
    payload: bytes

    def audio(self) -> np.ndarray:
        return np.frombuffer(self.payload, dtype="<i2")

    def text(self) -> str:
        return self.payload.decode(errors="replace")


class Decoder:
    """Incremental decoder: feed raw bytes, get complete frames. Resyncs on garbage."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[Frame]:
        self._buf += data
        frames = []
        while True:
            start = self._buf.find(MAGIC)
            if start < 0:
                del self._buf[:-1]  # keep a possible first magic byte
                return frames
            del self._buf[:start]
            if len(self._buf) < HEADER.size:
                return frames
            _, kind, _, seq, length = HEADER.unpack_from(self._buf)
            if kind not in (TYPE_INFO, TYPE_AUDIO) or length > 4096:
                del self._buf[:1]  # false magic, resync
                continue
            end = HEADER.size + length
            if len(self._buf) < end:
                return frames
            frames.append(Frame(kind, seq, bytes(self._buf[HEADER.size : end])))
            del self._buf[:end]


def encode(kind: int, seq: int, payload: bytes) -> bytes:
    """Inverse of the decoder (used for tests / simulation)."""
    return HEADER.pack(MAGIC, kind, 0, seq, len(payload)) + payload
