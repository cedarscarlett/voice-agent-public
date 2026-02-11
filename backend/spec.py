"""
SPEC-AS-CONSTANTS
-----------------
Single source of truth for all behavioral invariants in the system.

Rules:
- If changing a value changes runtime behavior, it belongs here.
- No magic numbers elsewhere in the codebase.
- Other modules MUST import from this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Tuple

# =============================================================================
# Audio Format (PCM16 mono @ 16kHz, 20ms frames)  [Spec §3]
# =============================================================================

AUDIO_SAMPLE_RATE_HZ: Final[int] = 16_000
AUDIO_CHANNELS: Final[int] = 1
AUDIO_SAMPLE_WIDTH_BYTES: Final[int] = 2  # PCM16 (signed 16-bit)
AUDIO_FRAME_MS: Final[int] = 20

AUDIO_SAMPLES_PER_FRAME: Final[int] = (AUDIO_SAMPLE_RATE_HZ * AUDIO_FRAME_MS) // 1000
AUDIO_BYTES_PER_FRAME_PCM: Final[int] = AUDIO_SAMPLES_PER_FRAME * AUDIO_SAMPLE_WIDTH_BYTES
AUDIO_FRAME_DURATION_S: Final[float] = AUDIO_FRAME_MS / 1000.0

# Browser capture is typically 48kHz; resample client-side in v1
BROWSER_CAPTURE_SAMPLE_RATE_HZ_DEFAULT: Final[int] = 48_000

# =============================================================================
# Binary WebSocket Frame Formats  [Spec §3, §15]
# =============================================================================
# Client → Server (mic audio): 4B seq_num + PCM frame
C2S_SEQ_NUM_BYTES: Final[int] = 4
C2S_FRAME_BYTES_TOTAL: Final[int] = C2S_SEQ_NUM_BYTES + AUDIO_BYTES_PER_FRAME_PCM

# Server → Client (TTS audio): 4B seq_num + 4B run_id + PCM frame
S2C_SEQ_NUM_BYTES: Final[int] = 4
S2C_RUN_ID_BYTES: Final[int] = 4
S2C_FRAME_BYTES_TOTAL: Final[int] = (
    S2C_SEQ_NUM_BYTES + S2C_RUN_ID_BYTES + AUDIO_BYTES_PER_FRAME_PCM
)

SEQ_NUM_START: Final[int] = 1
SEQ_NUM_MAX: Final[int] = 2**32 - 1  # u32 wraparound

# =============================================================================
# Session & State Timing  [Spec §4, §5, §12]
# =============================================================================

ERROR_STATE_AUTO_RESOLVE_DELAY_MS: Final[int] = 100
SESSION_RETAIN_AFTER_DISCONNECT_S: Final[float] = 5.0

# =============================================================================
# Endpointing / ASR Finalization  [Spec §8]
# =============================================================================

SILENCE_DETECTION_MS: Final[int] = 500
ASR_FINAL_TIMEOUT_MS: Final[int] = 800
ASR_PREFER_STABLE_PARTIAL_ON_TIMEOUT: Final[bool] = True

# =============================================================================
# Cancellation Protocol  [Spec §6, §18]
# =============================================================================

CANCEL_ACK_TIMEOUT_MS: Final[int] = 500

# =============================================================================
# TTS Chunking Policy  [Spec §7]
# =============================================================================

TTS_MIN_CHUNK_LEN_CHARS: Final[int] = 20
TTS_SIZE_TRIGGER_CHARS: Final[int] = 120
TTS_HARD_CAP_CHARS: Final[int] = 200
TTS_LOOKBACK_CHARS: Final[int] = 20
TTS_CHUNK_TIME_MS: Final[int] = 250
TTS_CHUNK_TIMER_POLL_MS: Final[int] = 50
TTS_MAX_CHUNK_CHARS: Final[int] = 500
PROVIDER_CHUNK_SIZE: Final[int] = 4096

# Preferred clause / punctuation boundaries
TTS_PREFERRED_BREAK_CHARS: Final[Tuple[str, ...]] = (
    ".", "!", "?", ",", ";", ":", "—", "\n"
)

# Safe fallback boundaries (word-aligned)
TTS_WHITESPACE_BREAK_CHARS: Final[Tuple[str, ...]] = (
    " ", "\t",
)

#TTS Concurrency
MAX_TTS_IN_FLIGHT: Final[int] = 1

# =============================================================================
# Conversation Context  [Spec §9]
# =============================================================================

MAX_CONTEXT_TURNS: Final[int] = 8
MAX_CONTEXT_CHARS: Final[int] = 6_000

# Truncation rule:
# While (turn_count > MAX_CONTEXT_TURNS) OR (total_chars > MAX_CONTEXT_CHARS):
#     drop oldest turn
#
# This enforces: keep the most recent turns that fit within BOTH constraints.

# =============================================================================
# Backpressure & DEGRADED Mode  [Spec §10]
# =============================================================================

# Hard queue capacities
INGEST_AUDIO_Q_MAX_S: Final[float] = 10.0 #TODO: back to 1.0
# TODO: Remove the audio out queue altogether
TTS_AUDIO_Q_MAX_S: Final[float] = 60.0

# DEGRADED entry thresholds (policy triggers)
DEGRADED_INGEST_AUDIO_Q_THRESHOLD_S: Final[float] = 1.0
DEGRADED_TTS_AUDIO_Q_THRESHOLD_S: Final[float] = 2.0

# DEGRADED exit thresholds
DEGRADED_EXIT_INGEST_AUDIO_Q_THRESHOLD_S: Final[float] = 0.5
DEGRADED_EXIT_TTS_AUDIO_Q_THRESHOLD_S: Final[float] = 0.5
DEGRADED_EXIT_SUSTAIN_MS: Final[int] = 500
DEGRADED_EXIT_CHECK_INTERVAL_MS: Final[int] = 100
DEGRADED_EXIT_REQUIRED_CONSECUTIVE_CHECKS: Final[int] = (
    DEGRADED_EXIT_SUSTAIN_MS // DEGRADED_EXIT_CHECK_INTERVAL_MS
)

ASR_EVENT_Q_MAX_EVENTS: Final[int] = 200
LLM_TOKEN_Q_MAX_CHARS: Final[int] = 20_000

# =============================================================================
# Failure Detection & Timeouts  [Spec §11]
# =============================================================================

LLM_FIRST_TOKEN_TIMEOUT_MS: Final[int] = 5_500 # TODO: change back to 1_500
TTS_FIRST_AUDIO_TIMEOUT_MS: Final[int] = 6000

LLM_STALL_TIMEOUT_S: Final[float] = 5.0
TTS_STALL_TIMEOUT_S: Final[float] = 8.0

# =============================================================================
# Retry Policy  [Spec §11]
# =============================================================================

INIT_RETRY_COUNT: Final[int] = 1
INIT_RETRY_DELAY_MS: Final[int] = 200

ASR_MAX_RETRIES: Final[int] = 3
ASR_RETRY_BACKOFF_MS: Final[Tuple[int, ...]] = (0, 200, 400, 800)

# =============================================================================
# Frontend Connection Backoff  [Spec §12]
# =============================================================================

CLIENT_WS_RETRY_BACKOFF_MS: Final[Tuple[int, ...]] = (200, 400, 800)

# =============================================================================
# Barge-in & Mic Gating  [Spec §8]
# =============================================================================

MIC_GATING_ENABLED_WHILE_SPEAKING: Final[bool] = True

BARGE_IN_HOLD_MS_MIN: Final[int] = 80
BARGE_IN_HOLD_MS_MAX: Final[int] = 150
BARGE_IN_RMS_THRESHOLD_DEFAULT: Final[float] = 0.08  # heuristic default

# =============================================================================
# LLM Prompt Versioning  [Spec §13]
# =============================================================================

SYSTEM_PROMPT_VERSION: Final[str] = "v1"
PROMPT_HASH_HEX_LEN: Final[int] = 8

# Note:
# - Actual prompt text lives in adapters/llm/prompts.py
# - This file only defines versioning + tracking invariants

# =============================================================================
# Observability  [Spec §14, §17]
# =============================================================================

# Per-turn metrics (authoritative list):
# - first_audio_frame_received_to_first_asr_partial_ms
# - asr_final_to_first_llm_token_ms
# - first_llm_token_to_first_tts_audio_ms
# - user_speech_end_to_audio_start_ms   (headline metric)
# - barge_in_signal_to_playback_stop_ms

QUEUE_DEPTH_LOG_INTERVAL_MS: Final[int] = 100

# =============================================================================
# Helper Functions
# =============================================================================

def frames_to_seconds(num_frames: int) -> float:
    """
    Convert a number of PCM frames to duration in seconds.

    Defensive behavior:
    - Negative input returns 0.0 instead of propagating an error.
    """
    if num_frames <= 0:
        return 0.0
    return num_frames * AUDIO_FRAME_DURATION_S


def seconds_to_frames(duration_s: float) -> int:
    """
    Convert a duration in seconds to whole PCM frames (floor).

    Defensive behavior:
    - Non-positive input returns 0.
    """
    if duration_s <= 0:
        return 0
    return int(duration_s / AUDIO_FRAME_DURATION_S)


# =============================================================================
# Convenience Bundles
# =============================================================================

@dataclass(frozen=True)
class AudioFormat:
    """
    Immutable bundle describing the PCM audio format.

    This is a convenience wrapper for passing format metadata around;
    it is NOT a second source of truth.
    """
    sample_rate_hz: int = AUDIO_SAMPLE_RATE_HZ
    channels: int = AUDIO_CHANNELS
    sample_width_bytes: int = AUDIO_SAMPLE_WIDTH_BYTES
    frame_ms: int = AUDIO_FRAME_MS

    @property
    def samples_per_frame(self) -> int:
        """Return number of samples per frame."""
        return (self.sample_rate_hz * self.frame_ms) // 1000

    @property
    def bytes_per_frame(self) -> int:
        """Return number of bytes per frame."""
        return self.samples_per_frame * self.sample_width_bytes


AUDIO_FORMAT_V1: Final[AudioFormat] = AudioFormat()

# =============================================================================
# Retry Delays
# =============================================================================

# ASR
ASR_RETRY_DELAYS_MS = [0, 200, 400, 800]

# LLM
LLM_INIT_RETRY_DELAY_MS = 300

# TTS
TTS_INIT_RETRY_DELAY_MS = 300
