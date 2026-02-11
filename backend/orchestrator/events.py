"""
Unified event definitions for the orchestrator reducer (v1).

Rules:
- Events describe facts that have occurred.
- Events carry data only (no behavior).
- All reducer decisions are based on these events.
- No clocks, no timers, no async, no side effects.
- Every event referenced in the spec/state table exists here.

Timer events are not ServiceEvents, but must carry run_id for stale gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from orchestrator.enums.service import Service


# =============================================================================
# Event Type Enumeration
# =============================================================================

class EventType(str, Enum):
    """
    Canonical event types understood by the reducer.

    Every (state, event_type) pair must be explicitly handled
    or explicitly ignored by the reducer.
    """

    # ------------------------------------------------------------------
    # Connection / session lifecycle
    # ------------------------------------------------------------------
    WS_CONNECTED = "WS_CONNECTED"
    WS_DISCONNECTED = "WS_DISCONNECTED"
    SESSION_STARTED = "SESSION_STARTED"
    SESSION_ENDED = "SESSION_ENDED"
    SESSION_END = "SESSION_END"

    # ------------------------------------------------------------------
    # Client control
    # ------------------------------------------------------------------
    MIC_START = "MIC_START"
    MIC_STOP = "MIC_STOP"
    STOP = "STOP"
    BARGE_IN = "BARGE_IN"
    USER_RETRY = "USER_RETRY"

    # ------------------------------------------------------------------
    # Audio ingest
    # ------------------------------------------------------------------
    AUDIO_FRAME = "AUDIO_FRAME"
    USER_SPEECH_DETECTED = "USER_SPEECH_DETECTED"

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------
    ASR_PARTIAL = "ASR_PARTIAL"
    ASR_FINAL = "ASR_FINAL"
    END_OF_SPEECH = "END_OF_SPEECH"
    ASR_ERROR = "ASR_ERROR"
    ASR_FINAL_TIMEOUT = "ASR_FINAL_TIMEOUT"

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    LLM_TOKEN = "LLM_TOKEN"
    LLM_DONE = "LLM_DONE"
    LLM_ERROR = "LLM_ERROR"
    LLM_FIRST_TOKEN_TIMEOUT = "LLM_FIRST_TOKEN_TIMEOUT"
    LLM_STALL_TIMEOUT = "LLM_STALL_TIMEOUT"

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------
    TTS_CHUNK_COMPLETE = "TTS_CHUNK_COMPLETE"
    TTS_AUDIO_FRAME = "TTS_AUDIO_FRAME"
    TTS_DONE = "TTS_DONE"
    TTS_ERROR = "TTS_ERROR"
    TTS_FIRST_AUDIO_TIMEOUT = "TTS_FIRST_AUDIO_TIMEOUT"
    TTS_STALL_TIMEOUT = "TTS_STALL_TIMEOUT"

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------
    CANCEL_ACK = "CANCEL_ACK"
    CANCEL_TIMEOUT = "CANCEL_TIMEOUT"

    # ------------------------------------------------------------------
    # Timers / internal control
    # ------------------------------------------------------------------
    CHUNK_TIMER_TICK = "CHUNK_TIMER_TICK"
    ERROR_TIMEOUT = "ERROR_TIMEOUT"

    # ------------------------------------------------------------------
    # Fatal
    # ------------------------------------------------------------------
    FATAL_ERROR = "FATAL_ERROR"

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------

    RETRY_READY = "retry_ready"

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------

    LLM_TOOL_CALL = "LLM_TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"

# =============================================================================
# Base Event
# =============================================================================

@dataclass(frozen=True)
class Event:
    """
    Base event type.

    All events must specify:
    - event_type: discriminant
    - ts_ms: timestamp provided by the source (or fake in tests)
    """

    event_type: EventType
    ts_ms: int


# =============================================================================
# Service-Scoped Events
# =============================================================================

@dataclass(frozen=True)
class ServiceEvent(Event):
    """
    Base class for events scoped to a versioned external service.

    The reducer MUST ignore events whose run_id does not match the
    currently active run for that service.
    """

    service: Service
    run_id: int


# =============================================================================
# Connection / Session Events
# =============================================================================

@dataclass(frozen=True)
class WSConnected(Event):
    """WebSocket connection established."""
    session_id: str


@dataclass(frozen=True)
class WSDisconnected(Event):
    """WebSocket connection lost."""
    session_id: str
    reason: str | None = None

@dataclass(frozen=True)
class SessionStarted(Event):
    """Session Started."""
    session_id: str

@dataclass(frozen=True)
class SessionEnded(Event):
    """Session Ended."""
    session_id: str


@dataclass(frozen=True)
class SessionEnd(Event):
    """Client requested graceful session termination."""


# =============================================================================
# Client Control Events
# =============================================================================

@dataclass(frozen=True)
class MicStart(Event):
    """Client started microphone capture."""


@dataclass(frozen=True)
class MicStop(Event):
    """Client stopped microphone capture."""


@dataclass(frozen=True)
class Stop(Event):
    """Client requested stop (cancel current operation)."""


@dataclass(frozen=True)
class BargeIn(Event):
    """Client detected barge-in during TTS playback."""


@dataclass(frozen=True)
class UserRetry(Event):
    """User manually clicked retry to exit ERROR state."""


# =============================================================================
# Audio Ingest
# =============================================================================

@dataclass(frozen=True)
class UserSpeechDetected(Event):
    """User speech has been detected."""

# =============================================================================
# ASR Events
# =============================================================================

@dataclass(frozen=True)
class ASRPartial(ServiceEvent):
    """
    Partial transcription result from ASR.

    May be unstable or revised by later partials.
    """
    text: str
    ts_range: tuple[int, int] | None = None  # (start_ms, end_ms)
    is_stable: bool = False


@dataclass(frozen=True)
class ASRFinal(ServiceEvent):
    """
    Final transcription result from ASR.

    This text is immutable and used as LLM input.
    """
    text: str
    ts_range: tuple[int, int] | None = None  # (start_ms, end_ms)


@dataclass(frozen=True)
class EndOfSpeech(ServiceEvent):
    """
    Endpoint detected by ASR or VAD.

    Indicates that user speech has ended and finalization
    should begin.
    """


@dataclass(frozen=True)
class ASRError(ServiceEvent):
    """
    ASR encountered an unrecoverable error during streaming.
    """
    reason: str


@dataclass(frozen=True)
class ASRFinalTimeout(ServiceEvent):
    """
    ASR_FINAL did not arrive within ASR_FINAL_TIMEOUT_MS after endpoint.
    The reducer must proceed using the best available partial.
    """



# =============================================================================
# LLM Events
# =============================================================================

@dataclass(frozen=True)
class LLMToken(ServiceEvent):
    """
    Streaming token delta produced by the LLM.
    """
    delta: str


@dataclass(frozen=True)
class LLMDone(ServiceEvent):
    """
    LLM completed generation successfully.
    """


@dataclass(frozen=True)
class LLMError(ServiceEvent):
    """
    LLM encountered an unrecoverable error.
    """
    reason: str


@dataclass(frozen=True)
class LLMFirstTokenTimeout(Event):
    """First LLM token did not arrive within timeout."""
    run_id: int


@dataclass(frozen=True)
class LLMStallTimeout(Event):
    """LLM stalled mid-stream."""
    run_id: int


# =============================================================================
# TTS Events
# =============================================================================

@dataclass(frozen=True)
class TTSChunkComplete(ServiceEvent):
    """
    TTS encountered an unrecoverable error.

    TTSChunkComplete is provider chunk completion (coarse)
    """
    chunk_index: int
    pcm_bytes: bytes


@dataclass(frozen=True)
class TTSAudioFrame(ServiceEvent):
    """
    One synthesized audio frame.
    """
    sequence_num: int
    pcm_bytes: bytes


@dataclass(frozen=True)
class TTSDone(ServiceEvent):
    """
    TTS completed synthesis successfully.
    """


@dataclass(frozen=True)
class TTSError(ServiceEvent):
    """
    TTS encountered an unrecoverable error.
    """
    reason: str


@dataclass(frozen=True)
class TTSFirstAudioTimeout(Event):
    """First TTS audio frame did not arrive within timeout."""
    run_id: int


@dataclass(frozen=True)
class TTSStallTimeout(Event):
    """TTS stalled mid-stream."""
    run_id: int


# =============================================================================
# Cancellation Events
# =============================================================================

@dataclass(frozen=True)
class CancelAck(ServiceEvent):
    """
    Acknowledgment that a Cancel(run_id) request has been observed.

    Note:
    - Emitted by the cancellation layer, not the service itself.
    - Inherits from ServiceEvent for reducer uniformity.
    """


@dataclass(frozen=True)
class CancelTimeout(Event):
    """
    Cancel acknowledgment did not arrive within timeout.

    Semantics:
    - Adapter must be hard-reset by runtime
    - Reducer proceeds without entering ERROR
    """
    service: Service
    run_id: int


# =============================================================================
# Timer / Internal Events
# =============================================================================

@dataclass(frozen=True)
class ChunkTimerTick(Event):
    """
    Periodic tick for TTS chunk timer evaluation.

    Injected by runtime every TTS_CHUNK_TIMER_POLL_MS while buffer is non-empty.
    """


@dataclass(frozen=True)
class ErrorTimeout(Event):
    """ERROR state auto-resolve delay elapsed."""


# =============================================================================
# Fatal
# =============================================================================

@dataclass(frozen=True)
class FatalError(Event):
    """Non-recoverable error forcing immediate teardown."""
    reason: str
    context: dict[str, Any] | None = None


# =============================================================================
# Retry
# =============================================================================

@dataclass(frozen=True)
class RetryReady(Event):
    """Emitted by runtime after retry delay expires."""
    service: Service
    system_prompt_version: str
    messages: tuple[dict[str, Any], ...]

# =============================================================================
# Tools
# =============================================================================

@dataclass(frozen=True)
class LLMToolCall(ServiceEvent):
    tool: str
    args: dict[str, Any]

@dataclass(frozen=True)
class ToolResult(Event):
    tool: str
    result: dict[str, Any]
