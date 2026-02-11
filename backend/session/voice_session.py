"""
Voice session container.

Per spec §17:
- Owns orchestrator state (immutable reducer pattern)
- Owns connection status (mutable, gateway-controlled)
- Owned and mutated by SessionGateway
- NOT a state machine
- Contains no orchestration logic
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from collections import deque


from context.conversation import ConversationContext
from orchestrator.runtime import Runtime
from session.connection_status import ConnectionStatus


# ---------------------------------------------------------------------
# VoiceSession
# ---------------------------------------------------------------------


@dataclass
class VoiceSession:
    """Mutable runtime container for a single voice session."""

    # ------------------------------------------------------------------
    # Identity / lifecycle
    # ------------------------------------------------------------------

    session_id: str
    created_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Conversation context (imperative, bounded)
    # ------------------------------------------------------------------

    conversation_context: ConversationContext = field(init=False)
    last_committed_turn_id: int = -1  # Sentinel for "no turns yet"

    # ------------------------------------------------------------------
    # Connection / gateway-controlled state
    # ------------------------------------------------------------------

    connection_status: ConnectionStatus = ConnectionStatus.DOWN
    websocket: Any = None  # Type: WebSocketProtocol in practice

    # ------------------------------------------------------------------
    # Runtime (executes commands + owns authoritative state)
    # ------------------------------------------------------------------

    runtime: Runtime | None = None

    # ------------------------------------------------------------------
    # Audio queues
    # ------------------------------------------------------------------

    audio_in_queue: Any = None  # Type: AudioIngestQueueProtocol in practice
    audio_out_queue: Any = None

    # ------------------------------------------------------------------
    # Service adapters (concrete, side-effectful)
    # ------------------------------------------------------------------

    asr_adapter: Any = None
    llm_adapter: Any = None
    tts_adapter: Any = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """
        Construct session-owned objects that require session_id.
        """
        # Conversation context is session-scoped
        self.conversation_context = ConversationContext(
            session_id=self.session_id
        )

        self._control_out: deque[dict[str, Any]] = deque()

    # ------------------------------------------------------------------
    # Wiring helpers (called by SessionGateway)
    # ------------------------------------------------------------------

    def attach_llm_adapter(self, adapter: Any) -> None:
        """
        Attach a concrete streaming LLM adapter.

        Called by SessionGateway during session bootstrap.
        Adapter must implement LLMAdapterProtocol.
        """
        self.llm_adapter = adapter

    def attach_asr_adapter(self, adapter: Any) -> None:
        """Attach a concrete streaming ASR adapter."""
        self.asr_adapter = adapter

    def attach_tts_adapter(self, adapter: Any) -> None:
        """Attach a concrete streaming TTS adapter."""
        self.tts_adapter = adapter

    def attach_runtime(self, runtime: Runtime) -> None:
        """
        Attach the runtime executor.

        Called by SessionGateway during session bootstrap.
        Must be called after adapters are attached.
        """
        self.runtime = runtime

    # ------------------------------------------------------------------
    # Observability helpers (read-only)
    # ------------------------------------------------------------------


    def log_context(self) -> dict[str, Any]:
        """
        Return standard logging context for this session (spec §14).

        Intended for gateway / observability enrichment.
        """
        return {
            "session_id": self.session_id,
            "connection_status": self.connection_status.value,
        }

    def enqueue_control(self, msg: dict[str, Any]) -> None:
        """
        Enqueue a control message for gateway delivery to the client.

        Messages are buffered in FIFO order and later retrieved via
        drain_control().
        """
        self._control_out.append(msg)

    def drain_control(self) -> tuple[dict[str, Any], ...]:
        """
        Atomically drain all pending control messages.

        Returns:
            A FIFO-ordered tuple of control messages. Returns an empty
            tuple if no messages are pending.

        After this call, the control queue is empty.
        """
        if not self._control_out:
            return ()
        out = tuple(self._control_out)
        self._control_out.clear()
        return out
