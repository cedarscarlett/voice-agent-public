# backend/protocol/binary.py
"""
Binary framing helpers for audio transport.

Spec §3, §6, §15:
- Client → Server (mic):
    4 bytes  seq_num (u32, little-endian)
    640 bytes PCM16 audio

- Server → Client (TTS):
    4 bytes  seq_num (u32, little-endian)
    4 bytes  run_id  (u32, little-endian)
    640 bytes PCM16 audio

Usage example:

    frame = decode_c2s_frame(payload, ts_ms=now_ms)

    result = check_sequence_gap(last_seq=prev_seq, current_seq=frame.sequence_num)
    if result.gap:
        log_event({
            "event_type": "seq_gap_detected",
            "expected": result.expected,
            "actual": result.actual,
            "gap_size": result.gap_size,
        })

    payload = encode_s2c_frame(
        sequence_num=tts_seq,
        run_id=active_tts_run_id,
        pcm_bytes=frame.pcm_bytes,
    )
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

from audio.frames import AudioFrame
from spec import (
    AUDIO_BYTES_PER_FRAME_PCM,
    C2S_FRAME_BYTES_TOTAL,
    S2C_FRAME_BYTES_TOTAL,
    SEQ_NUM_START,
    SEQ_NUM_MAX,
)


# -------------------------
# Exceptions
# -------------------------

class BinaryProtocolError(Exception):
    """Base class for binary protocol errors."""


class InvalidFrameLength(BinaryProtocolError):
    """
    Raised when a binary audio frame does not match the expected byte length.

    Indicates a violation of the binary framing contract (truncated, oversized,
    or malformed payload). The frame is unsafe to process and must be dropped.
    """


class InvalidSequenceNumber(BinaryProtocolError):
    """
    Raised when a sequence number or run_id is outside the valid range.

    Indicates a protocol violation that would break ordering, gap detection,
    or stale-frame filtering.
    """


# -------------------------
# Low-level helpers
# -------------------------

def _u32_le(value: int) -> bytes:
    return struct.pack("<I", value)


def _read_u32_le(buf: bytes, offset: int = 0) -> int:
    return struct.unpack_from("<I", buf, offset)[0]


def is_seq_next(prev: int, current: int) -> bool:
    """
    Return True if `current` is the expected next sequence number
    after `prev`, accounting for wraparound.
    """
    if prev == SEQ_NUM_MAX:
        return current == SEQ_NUM_START
    return current == prev + 1


# -------------------------
# Client → Server (mic)
# -------------------------

def decode_c2s_frame(payload: bytes, *, ts_ms: int) -> AudioFrame:
    """
    Decode a client→server mic audio frame.
    """
    if len(payload) != C2S_FRAME_BYTES_TOTAL:
        raise InvalidFrameLength(
            f"C2S frame length {len(payload)} != {C2S_FRAME_BYTES_TOTAL}"
        )

    seq = _read_u32_le(payload, 0)

    if seq < SEQ_NUM_START or seq > SEQ_NUM_MAX:
        raise InvalidSequenceNumber(f"Invalid seq_num: {seq}")

    pcm_bytes = payload[4:]

    if len(pcm_bytes) != AUDIO_BYTES_PER_FRAME_PCM:
        raise InvalidFrameLength(
            f"PCM length {len(pcm_bytes)} != {AUDIO_BYTES_PER_FRAME_PCM}"
        )

    return AudioFrame(
        sequence_num=seq,
        pcm_bytes=pcm_bytes,
        ts_ms=ts_ms,
    )


# -------------------------
# Server → Client (TTS)
# -------------------------

def encode_s2c_frame(
    *,
    sequence_num: int,
    run_id: int,
    pcm_bytes: bytes,
) -> bytes:
    """
    Encode a server→client TTS audio frame.
    """
    if sequence_num < SEQ_NUM_START or sequence_num > SEQ_NUM_MAX:
        raise InvalidSequenceNumber(f"Invalid seq_num: {sequence_num}")

    # Spec §6: run_ids start at 1; 0 is internal-only
    if run_id < 1:
        raise InvalidSequenceNumber(f"Invalid run_id: {run_id}")

    if len(pcm_bytes) != AUDIO_BYTES_PER_FRAME_PCM:
        raise InvalidFrameLength(
            f"PCM length {len(pcm_bytes)} != {AUDIO_BYTES_PER_FRAME_PCM}"
        )

    payload = (
        _u32_le(sequence_num)
        + _u32_le(run_id)
        + pcm_bytes
    )

    if len(payload) != S2C_FRAME_BYTES_TOTAL:
        raise InvalidFrameLength(
            f"S2C frame length {len(payload)} != {S2C_FRAME_BYTES_TOTAL}"
        )

    return payload


# -------------------------
# Sequence gap detection
# -------------------------

@dataclass(frozen=True)
class SeqCheckResult:
    """
    Result of a sequence continuity check.
    """
    gap: bool
    expected: int
    actual: int

    @property
    def gap_size(self) -> int:
        """
        Number of frames skipped (0 if no gap).

        Handles wraparound correctly.
        """
        if not self.gap:
            return 0

        # Linear (no wrap)
        if self.actual > self.expected:
            return self.actual - self.expected

        # Wraparound
        return (SEQ_NUM_MAX - self.expected + 1) + (self.actual - SEQ_NUM_START)


def check_sequence_gap(
    *,
    last_seq: Optional[int],
    current_seq: int,
) -> SeqCheckResult:
    """
    Check whether `current_seq` follows `last_seq`.

    Pure function; never raises.
    """
    if last_seq is None:
        return SeqCheckResult(
            gap=False,
            expected=current_seq,
            actual=current_seq,
        )

    if is_seq_next(last_seq, current_seq):
        return SeqCheckResult(
            gap=False,
            expected=current_seq,
            actual=current_seq,
        )

    expected = SEQ_NUM_START if last_seq == SEQ_NUM_MAX else last_seq + 1

    return SeqCheckResult(
        gap=True,
        expected=expected,
        actual=current_seq,
    )
