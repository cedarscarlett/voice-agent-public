"""
Authoritative orchestrator state container.

Rules:
- This dataclass is a pure data model.
- It contains ALL state the reducer may ever need.
- No behavior, no helpers, no derived logic.
- Fields may exist unused in early phases (by design).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from orchestrator.enums.state import State
from orchestrator.enums.mode import Mode
from orchestrator.enums.service import Service
from orchestrator.run_ids import RunIds
from orchestrator.retry import RetryAttempt

from context.conversation import ConversationContext


# =============================================================================
# Conversation / Turns
# =============================================================================

@dataclass(frozen=True)
class Turn:
    """Single conversation turn."""
    role: Literal["user", "assistant"]
    text: str
    turn_id: int


# =============================================================================
# TTS Chunk Timer State
# =============================================================================

@dataclass(frozen=True)
class ChunkTimerState:
    """
    Pure timer state for TTS chunking (token-driven streaming).

    Timer lifecycle:
    - Starts when buffer transitions from empty → non-empty
    - Elapsed computed from event timestamps
    - Reset when chunk sent or B/C triggers fire without sending
    """
    active: bool = False
    start_ts_ms: int = 0


# =============================================================================
# Orchestrator State
# =============================================================================

@dataclass(frozen=True)
class OrchestratorState:
    """Immutable snapshot of all orchestrator-owned state."""

    # or remove the default and make gateway construct this with a
    # voice received from config

    # asr_authoritative:
    # True iff ASR output is semantically meaningful and may be admitted.
    # Independent of ASR process existence or run_id monotonicity.
    # Controlled exclusively by reducer turn events.
    conversation_context: ConversationContext

    # --- Streaming TTS (Phase: migrate SPEAKING -> PROCESSING) ---

    pending_tts_chunks: tuple[str, ...] = ()

    tts_next_play_chunk_index: int = 0

    tts_ready_audio_chunks: dict[int, tuple[bytes, ...]] = field(
    default_factory=lambda: {}
)

    # True after first StartTTS in assistant turn
    tts_run_started_for_turn: bool = False

    # True after LLMDone for the current assistant turn
    llm_finished: bool = False

    # Count of outstanding StartTTS requests not yet completed/errored
    tts_in_flight: int = 0

    # Monotonic chunk_index for StartTTS within the current assistant turn
    tts_chunk_index: int = 0

    # Optimistically accumulated assistant text (append at StartTTS emission)
    assistant_text_accum: str = ""


    asr_authoritative: bool = False

    tts_voice: str = "sarah"

    # ------------------------------------------------------------------
    # Control state
    # ------------------------------------------------------------------
    state: State = State.IDLE
    mode: Mode = Mode.NORMAL

    # ------------------------------------------------------------------
    # Run/version tracking
    # ------------------------------------------------------------------
    active_runs: RunIds = field(default_factory=RunIds)
    cancel_in_flight: frozenset[Service] = frozenset()

    # ------------------------------------------------------------------
    # ASR / endpointing
    # ------------------------------------------------------------------
    asr_endpoint_ts_ms: int | None = None

    # ------------------------------------------------------------------
    # LLM → TTS chunking (text accumulation + timer)
    # ------------------------------------------------------------------
    tts_text_buffer: str = ""
    chunk_timer_state: ChunkTimerState = field(default_factory=ChunkTimerState)

    # ------------------------------------------------------------------
    # TTS chunk execution (Phase 8)
    # ------------------------------------------------------------------
    # Frozen chunk plan generated on LLMDone
    tts_chunks: tuple[str, ...] = ()

    # Index of the chunk currently being synthesized
    tts_current_chunk_index: int = 0

    # Used for first-audio timeout + latency metrics
    tts_first_audio_received: bool = False

    # ------------------------------------------------------------------
    # Conversation context
    # ------------------------------------------------------------------

    current_turn_id: int = 0
    pending_user_turn_text: str = ""

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------
    last_error: str | None = None

    # ------------------------------------------------------------------
    # Backpressure / DEGRADED mode (future)
    # ------------------------------------------------------------------
    # queue_depths_s: QueueDepths | None = None

    # ------------------------------------------------------------------
    # Retry bookkeeping (LLM init)
    # ------------------------------------------------------------------
    # Supersedes earlier plan field `llm_init_attempts: int`.
    llm_retry_attempt: RetryAttempt = RetryAttempt(attempt=0)

    # Used to distinguish init vs mid-stream failures in reducer.
    # Set True on first LLMToken for the active run.
    llm_first_token_received: bool = False
