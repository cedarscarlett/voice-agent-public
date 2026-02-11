# tests/unit/test_frame_generator.py

from audio.frame_generator import split_pcm_into_frames
from spec import AUDIO_BYTES_PER_FRAME_PCM


def test_correct_bytes_per_frame():
    # 3 full frames
    pcm = b"\x00" * (AUDIO_BYTES_PER_FRAME_PCM * 3)

    frames = split_pcm_into_frames(pcm)

    assert len(frames) == 3
    for frame in frames:
        assert len(frame) == AUDIO_BYTES_PER_FRAME_PCM


def test_drops_incomplete_trailing_frame():
    pcm = b"\x00" * (AUDIO_BYTES_PER_FRAME_PCM * 2 + 10)

    frames = split_pcm_into_frames(pcm)

    assert len(frames) == 2
    for frame in frames:
        assert len(frame) == AUDIO_BYTES_PER_FRAME_PCM


def test_empty_input_returns_no_frames():
    frames = split_pcm_into_frames(b"")
    assert frames == []
