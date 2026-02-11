"""
Runtime execution context.

Provides Runtime with live access to session-owned imperative resources
needed for command execution and side effects (adapters, queues, status).

This module contains:
- Narrow Protocols (capabilities, not implementations)
- Zero orchestration logic
- Zero state mutation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable, Any
from session.connection_status import ConnectionStatus

if TYPE_CHECKING:
    from session.voice_session import VoiceSession
    from audio.frames import AudioFrame


# ---------------------------------------------------------------------
# Adapter Protocols
# ---------------------------------------------------------------------

@runtime_checkable
class ASRAdapterProtocol(Protocol):
    async def start_stream(self, run_id: int) -> None: ...
    async def send_audio(
        self,
        run_id: int,
        sequence_num: int,
        pcm_bytes: bytes,
    ) -> None: ...
    async def cancel(self, run_id: int) -> None: ...
    def force_reset(self) -> None:
        """
        Hard reset of ASR connection (e.g., after timeout exhaustion).
        May tear down and recreate internal state.
        """


@runtime_checkable
class LLMAdapterProtocol(Protocol):
    async def start_completion(
        self,
        *,
        run_id: int,
        messages: list[dict[str, str]],
    ) -> None: ...

    async def cancel(self, run_id: int) -> None: ...
    def force_reset(self) -> None:
        """
        Hard reset of LLM connection.
        """


@runtime_checkable
class TTSAdapterProtocol(Protocol):
    """
    Chunk-oriented TTS adapter protocol.

    Contract:
    - synthesize_chunk() must synthesize exactly one chunk
    - Adapter must emit exactly one terminal event:
        - TTSChunkComplete(run_id, chunk_index, pcm_bytes)
        - OR TTSError(run_id, reason)
    - Adapter is stateless across chunks except for run_id scoping
    """

    async def synthesize_chunk(
        self,
        *,
        run_id: int,
        chunk_index: int,
        text: str,
        voice: str
    ) -> None: ...

    async def cancel(self, run_id: int) -> None: ...

    def force_reset(self) -> None:
        """
        Hard reset of TTS connection (e.g., after timeout exhaustion).
        """


# ---------------------------------------------------------------------
# Audio Queue Protocols
# ---------------------------------------------------------------------

class AudioFrameProtocol(Protocol):
    sequence_num: int
    pcm_bytes: bytes


class AudioInQueueProtocol(Protocol):
    def dequeue(self) -> AudioFrameProtocol | None: ...
    def depth_seconds(self) -> float: ...
    def peek(self) -> AudioFrame | None: ...
    def peek_latest(self) -> AudioFrame | None: ...


# ---------------------------------------------------------------------
# Runtime Execution Context
# ---------------------------------------------------------------------

class RuntimeExecutionContext:
    """
    Imperative execution context for Runtime.

    This object provides *live views* into session-owned resources
    so Runtime does not need to synchronize or cache anything.

    Runtime is allowed to:
    - Call adapters
    - Read queues
    - Observe connection state

    Runtime is NOT allowed to:
    - Mutate session state directly
    - Perform orchestration decisions
    """

    def __init__(self, session: VoiceSession) -> None:
        self.session = session

    # ----------------------------
    # Session metadata
    # ----------------------------

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def connection_status(self) -> ConnectionStatus:
        return self.session.connection_status

    # ----------------------------
    # Adapters
    # ----------------------------

    @property
    def asr_adapter(self) -> ASRAdapterProtocol | None:
        return self.session.asr_adapter

    @property
    def llm_adapter(self) -> LLMAdapterProtocol | None:
        return self.session.llm_adapter

    @property
    def tts_adapter(self) -> TTSAdapterProtocol | None:
        return self.session.tts_adapter

    # ----------------------------
    # Audio ingress
    # ----------------------------

    @property
    def audio_in_queue(self) -> AudioInQueueProtocol | None:
        # AudioFrameQueue already conforms structurally.
        q: Any = self.session.audio_in_queue
        return q
