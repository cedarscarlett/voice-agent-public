"""
PCM frame splitting utilities (pure).

Purpose:
- Convert a provider-produced PCM blob into fixed-size 20ms PCM16 frames
  for enqueueing into AudioFrameQueue and transport over the WS binary protocol.

Invariants (Spec §3):
- PCM16 signed, little-endian
- Mono
- 16 kHz
- 20 ms frames
- Bytes per frame = spec.AUDIO_BYTES_PER_FRAME_PCM

Design:
- Pure functions only (no queues, no timing, no IO).
- Defensive validation to catch format drift early.
- Drops any incomplete trailing frame (Spec Phase 8.7).
"""

from __future__ import annotations

from spec import (
    AUDIO_BYTES_PER_FRAME_PCM,
    AUDIO_FRAME_MS,
    AUDIO_SAMPLE_RATE_HZ,
    AUDIO_SAMPLE_WIDTH_BYTES,
    AUDIO_CHANNELS,
)


def split_pcm_into_frames(
    pcm_bytes: bytes,
    *,
    sample_rate_hz: int = AUDIO_SAMPLE_RATE_HZ,
    frame_duration_ms: int = AUDIO_FRAME_MS,
    channels: int = AUDIO_CHANNELS,
    sample_width_bytes: int = AUDIO_SAMPLE_WIDTH_BYTES,
) -> list[bytes]:
    """
    Split raw PCM16 bytes into fixed-size frames.

    Args:
        pcm_bytes:
            Raw PCM bytes. Expected to be PCM16 (signed) little-endian.
        sample_rate_hz:
            Sample rate of the PCM stream. v1 expects 16_000.
        frame_duration_ms:
            Frame size in milliseconds. v1 expects 20.
        channels:
            Number of channels. v1 expects mono (1).
        sample_width_bytes:
            Bytes per sample. v1 expects 2 for PCM16.

    Returns:
        List of frame byte strings. Each frame is exactly the computed frame size.
        Drops any incomplete trailing frame.

    Raises:
        ValueError if parameters are invalid or inconsistent.

    Notes:
        This function does NOT:
        - validate WAV headers (input must already be raw PCM)
        - resample audio
        - pad trailing audio
    """
    print(
    "[TTS SPLIT DEBUG]",
    {
        "pcm_len": len(pcm_bytes),
        "bytes_per_frame": AUDIO_BYTES_PER_FRAME_PCM,
        "mod": len(pcm_bytes) % AUDIO_BYTES_PER_FRAME_PCM,
    }
)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    if frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be > 0")
    if channels <= 0:
        raise ValueError("channels must be > 0")
    if sample_width_bytes <= 0:
        raise ValueError("sample_width_bytes must be > 0")

    # Compute bytes per frame from parameters
    samples_per_frame = (sample_rate_hz * frame_duration_ms) // 1000
    bytes_per_frame = samples_per_frame * channels * sample_width_bytes

    # v1 invariant check: ensure computed frame size matches spec constant
    # (prevents silent drift if a caller passes non-v1 parameters).
    if (
        sample_rate_hz == AUDIO_SAMPLE_RATE_HZ
        and frame_duration_ms == AUDIO_FRAME_MS
        and channels == AUDIO_CHANNELS
        and sample_width_bytes == AUDIO_SAMPLE_WIDTH_BYTES
        and bytes_per_frame != AUDIO_BYTES_PER_FRAME_PCM
    ):
        raise ValueError(
            "Computed bytes_per_frame does not match spec.AUDIO_BYTES_PER_FRAME_PCM "
            f"(computed={bytes_per_frame}, spec={AUDIO_BYTES_PER_FRAME_PCM})"
        )

    if bytes_per_frame <= 0:
        raise ValueError("bytes_per_frame must be > 0")

    # Fast-path: empty input
    if not pcm_bytes:
        print("TTS SPLIT returning due to empty input")
        return []

    # Drop incomplete trailing frame (no padding).
    whole_frames = len(pcm_bytes) // bytes_per_frame
    if whole_frames <= 0:
        print("TTS SPLIT returning due to incomplete trailing frame")
        return []

    out: list[bytes] = []
    end = whole_frames * bytes_per_frame
    for offset in range(0, end, bytes_per_frame):
        out.append(pcm_bytes[offset : offset + bytes_per_frame])

    print(
        "[TTS SPLIT RESULT]",
        {
            "frames": len(out),
            "total_pcm_len": len(pcm_bytes),
        }
    )

    return out


def bytes_to_frame_count(
    num_bytes: int,
    *,
    sample_rate_hz: int = AUDIO_SAMPLE_RATE_HZ,
    frame_duration_ms: int = AUDIO_FRAME_MS,
    channels: int = AUDIO_CHANNELS,
    sample_width_bytes: int = AUDIO_SAMPLE_WIDTH_BYTES,
) -> int:
    """
    Return the number of whole frames represented by num_bytes.

    Drops any incomplete trailing frame (floor division).
    Useful for observability and tests.

    Raises:
        ValueError if parameters are invalid.
    """
    if num_bytes <= 0:
        return 0

    samples_per_frame = (sample_rate_hz * frame_duration_ms) // 1000
    bytes_per_frame = samples_per_frame * channels * sample_width_bytes
    if bytes_per_frame <= 0:
        raise ValueError("bytes_per_frame must be > 0")

    return num_bytes // bytes_per_frame
