"""
Side-effect command definitions for the orchestrator (v2).

Rules:
- Commands are declarative requests for side effects.
- Commands are emitted by the reducer and executed by the runtime.
- No behavior, no async, no I/O, no clocks.
- Reducer logic remains pure and deterministic.
Invariant:
    - All concrete Command subclasses MUST be frozen dataclasses.
    - Commands are immutable value objects emitted by the reducer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from orchestrator.events import EventType
from orchestrator.enums.service import Service

# =============================================================================
# Command Type Enumeration
# =============================================================================

class CommandType(str, Enum):
    """
    Canonical command types emitted by the reducer.

    These are stable discriminants used for logging, replay,
    and runtime dispatch.
    """

    # ASR
    START_ASR = "START_ASR"
    STOP_ASR = "STOP_ASR"
    SEND_ASR_AUDIO = "SEND_ASR_AUDIO"

    # LLM
    START_LLM = "START_LLM"
    CANCEL_LLM = "CANCEL_LLM"

    # TTS
    START_TTS = "START_TTS"
    SEND_TTS_TEXT = "SEND_TTS_TEXT"
    CANCEL_TTS = "CANCEL_TTS"

    # Client / transport
    SEND_JSON_TO_CLIENT = "SEND_JSON_TO_CLIENT"
    SEND_AUDIO_TO_CLIENT = "SEND_AUDIO_TO_CLIENT"

    # Session / lifecycle
    END_SESSION = "END_SESSION"

    # Timers
    START_TIMER = "START_TIMER"
    CANCEL_TIMER = "CANCEL_TIMER"

    # Observability
    LOG_EVENT = "LOG_EVENT"
    RECORD_METRIC = "RECORD_METRIC"

    # Tools
    EXECUTE_TOOL =" EXECUTE_TOOL"


# =============================================================================
# Base Command
# =============================================================================

class Command:
    """
    Base command type.

    command_type is an explicit discriminant and must never be inferred
    from Python type identity.
    """

    command_type: CommandType


# =============================================================================
# ASR Commands
# =============================================================================

@dataclass(frozen=True)
class StartASR(Command):
    """Request to start a new ASR stream."""
    run_id: int
    audio_format: str = "pcm16_16khz_mono"
    command_type: CommandType = CommandType.START_ASR


@dataclass(frozen=True)
class StopASR(Command):
    """Request to stop the active ASR stream."""
    run_id: int
    command_type: CommandType = CommandType.STOP_ASR


# =============================================================================
# LLM Commands
# =============================================================================

@dataclass(frozen=True)
class StartLLM(Command):
    """
    Request to start a new LLM completion.

    messages must be immutable; reducer owns serialization.
    """
    run_id: int
    system_prompt_version: str
    messages: tuple[dict[str, Any], ...]
    command_type: CommandType = CommandType.START_LLM


@dataclass(frozen=True)
class CancelLLM(Command):
    """Request to cancel an in-flight LLM run."""
    run_id: int
    command_type: CommandType = CommandType.CANCEL_LLM

@dataclass(frozen=True)
class CommitTurn(Command):
    """Commit completed turn to conversation context."""
    turn_id: int
    user_text: str
    assistant_text: str


# =============================================================================
# TTS Commands
# =============================================================================

@dataclass(frozen=True)
class StartTTS(Command):
    """
    Request to synthesize a single TTS chunk.

    The adapter must emit exactly one TTSChunkComplete or TTSError
    in response.
    """
    run_id: int
    chunk_index: int
    text: str
    voice: str
    command_type: CommandType = CommandType.START_TTS


@dataclass(frozen=True)
class SendTTSText(Command):
    """Request to send a text chunk to TTS."""
    run_id: int
    text: str
    command_type: CommandType = CommandType.SEND_TTS_TEXT


@dataclass(frozen=True)
class CancelTTS(Command):
    """Request to cancel an in-flight TTS run."""
    run_id: int
    command_type: CommandType = CommandType.CANCEL_TTS

@dataclass(frozen=True)
class EnqueueAudioFrames(Command):
    """Request to enqueue PCM audio frames for downstream audio output."""
    run_id: int
    frames: tuple[bytes, ...]  # each is 20ms PCM16 frame



# =============================================================================
# Client / Transport Commands
# =============================================================================

@dataclass(frozen=True)
class SendJSONToClient(Command):
    """
    Send a JSON control or UI message to the client.
    """
    message_type: str
    data: dict[str, Any]
    command_type: CommandType = CommandType.SEND_JSON_TO_CLIENT


@dataclass(frozen=True)
class SendAudioToClient(Command):
    """
    Send a TTS audio frame to the client.
    """
    run_id: int
    sequence_num: int
    pcm_bytes: bytes
    command_type: CommandType = CommandType.SEND_AUDIO_TO_CLIENT


# =============================================================================
# Session / Lifecycle Commands
# =============================================================================

@dataclass(frozen=True)
class EndSession(Command):
    """Request graceful session termination."""
    reason: str | None = None
    command_type: CommandType = CommandType.END_SESSION


# =============================================================================
# Timer Commands
# =============================================================================

@dataclass(frozen=True)
class StartTimer(Command):
    """
    Request to start a named timer.

    On expiration, the runtime must inject the specified timeout event.
    """
    timer_id: str
    duration_ms: int
    timeout_event_type: EventType
    command_type: CommandType = CommandType.START_TIMER


@dataclass(frozen=True)
class CancelTimer(Command):
    """Request to cancel a previously scheduled timer."""
    timer_id: str
    command_type: CommandType = CommandType.CANCEL_TIMER


# =============================================================================
# Observability Commands
# =============================================================================

@dataclass(frozen=True)
class LogEvent(Command):
    """Request to emit a structured observability event."""
    event: dict[str, Any]
    command_type: CommandType = CommandType.LOG_EVENT


@dataclass(frozen=True)
class RecordMetric(Command):
    """Request to record a metric value."""
    name: str
    value: float
    tags: tuple[tuple[str, str], ...] | None = None
    command_type: CommandType = CommandType.RECORD_METRIC


# =============================================================================
# Retry Commands
# =============================================================================

@dataclass(frozen=True)
class ScheduleRetry(Command):
    """
    Request that runtime schedule a retry attempt after delay_ms.

    Runtime responsibilities:
    - wait delay_ms
    - emit RetryReady(service=..., system_prompt_version=..., messages=...)

    Reducer remains pure: it decides *that* a retry should happen,
    runtime performs the waiting.
    """
    service: Service
    delay_ms: int
    system_prompt_version: str
    messages: tuple[dict[str, Any], ...]  # Frozen conversation context

# =============================================================================
# Tool Commands
# =============================================================================

@dataclass(frozen=True)
class ExecuteTool(Command):
    tool: str
    args: dict[str, Any]
