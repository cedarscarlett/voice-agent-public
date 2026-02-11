# pylint: disable=missing-module-docstring,missing-function-docstring

import pytest

from protocol.binary import (
    decode_c2s_frame,
    encode_s2c_frame,
    check_sequence_gap,
    InvalidFrameLength,
    InvalidSequenceNumber,
)
from spec import (
    AUDIO_BYTES_PER_FRAME_PCM,
    SEQ_NUM_MAX,
    SEQ_NUM_START,
)


def make_valid_pcm() -> bytes:
    return b"\x00\x00" * (AUDIO_BYTES_PER_FRAME_PCM // 2)


# ---------------------------------------------------------------------
# Invalid frame lengths
# ---------------------------------------------------------------------

def test_decode_rejects_short_frame():
    payload = b"\x01\x00\x00\x00" + make_valid_pcm()[:-2]

    with pytest.raises(InvalidFrameLength):
        decode_c2s_frame(payload, ts_ms=123)


def test_decode_rejects_long_frame():
    payload = b"\x01\x00\x00\x00" + make_valid_pcm() + b"\x00\x00"

    with pytest.raises(InvalidFrameLength):
        decode_c2s_frame(payload, ts_ms=123)


# ---------------------------------------------------------------------
# Sequence number validation
# ---------------------------------------------------------------------

def test_decode_rejects_seq_zero():
    payload = (0).to_bytes(4, "little") + make_valid_pcm()

    with pytest.raises(InvalidSequenceNumber):
        decode_c2s_frame(payload, ts_ms=123)


def test_encode_rejects_invalid_seq():
    with pytest.raises(InvalidSequenceNumber):
        encode_s2c_frame(
            sequence_num=0,
            run_id=1,
            pcm_bytes=make_valid_pcm(),
        )


def test_encode_rejects_invalid_run_id():
    with pytest.raises(InvalidSequenceNumber):
        encode_s2c_frame(
            sequence_num=1,
            run_id=0,
            pcm_bytes=make_valid_pcm(),
        )


# ---------------------------------------------------------------------
# Sequence gap detection
# ---------------------------------------------------------------------

def test_sequence_gap_detected():
    result = check_sequence_gap(last_seq=5, current_seq=8)

    assert result.gap is True
    assert result.expected == 6
    assert result.actual == 8
    assert result.gap_size == 2


def test_sequence_no_gap():
    result = check_sequence_gap(last_seq=5, current_seq=6)

    assert result.gap is False
    assert result.gap_size == 0


# ---------------------------------------------------------------------
# Wraparound handling
# ---------------------------------------------------------------------

def test_sequence_wraparound_no_gap():
    result = check_sequence_gap(last_seq=SEQ_NUM_MAX, current_seq=SEQ_NUM_START)

    assert result.gap is False


def test_sequence_wraparound_gap():
    result = check_sequence_gap(last_seq=SEQ_NUM_MAX - 2, current_seq=1)

    assert result.gap is True
    assert result.gap_size == 2
