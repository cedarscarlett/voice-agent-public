"""
Audio frame primitives.

Pure data containers only.
No behavior, no queues, no timing logic.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioFrame:
    """
    Canonical audio frame used throughout the backend audio pipeline.

    sequence_num:
        Monotonic sequence number provided by the sender (client or TTS).
        Used for gap detection and debugging only.

    pcm_bytes:
        Raw PCM16 audio bytes.
        Length MUST equal spec.AUDIO_BYTES_PER_FRAME_PCM.

    ts_ms:
        Wall-clock timestamp (milliseconds) when the frame was received
        or produced. Used for observability only (not control logic).
    """
    sequence_num: int
    pcm_bytes: bytes
    ts_ms: int
    run_id: int = 0  # 0 = client→server (mic), >0 = server→client (TTS)
