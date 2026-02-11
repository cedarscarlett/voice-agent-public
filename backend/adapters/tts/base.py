"""
TTS adapter contract (Phase 8).

This module defines the *interface only* — no chunking policy, no frame splitting,
no retries, timers, or orchestration decisions live here.

Key invariants (spec §8, §17, §18):
- Chunking is orchestrator-owned and deterministic. TTS adapters receive pre-chunked
  text and must not implement independent chunking logic.
- Run IDs are owned by the orchestrator (monotonic per service). Adapters never
  generate or mutate run IDs.
- The adapter emits TTS events; it does not call the reducer or make state
  transitions.
- Cancellation is explicit: cancel(run_id) is a request to stop producing output
  for that run as quickly as possible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TTSAdapter(ABC):
    """
    Abstract interface for a chunked (non-streaming) TTS adapter.

    Note: emit_event callback must be async.

    Implementations are responsible for:
    - Accepting pre-chunked text for a specific run_id via synthesize_chunk()
    - Calling the TTS provider and converting response to PCM16 16kHz mono
    - Producing TTS events (chunk_complete/error) for that run_id
    - Supporting cancellation via cancel()

    Non-responsibilities:
    - No chunking policy decisions (text is pre-chunked by orchestrator)
    - No frame splitting (orchestrator converts PCM blobs to 20ms frames)
    - No state machine logic (IDLE/PROCESSING/SPEAKING/etc.)
    - No timers owned by the reducer
    - No direct interaction with WebSocket or UI
    """

    @abstractmethod
    async def synthesize_chunk(
        self,
        *,
        run_id: int,
        chunk_index: int,
        text: str,
        voice: str,
    ) -> None:
        """
        Synthesize a single pre-chunked text segment.

        The adapter will emit TTS events asynchronously via the configured
        event sink (callback or queue).

        Args:
            run_id: Orchestrator-owned run identifier for the current TTS stream.
            chunk_index: Index of this chunk within the full response (for ordering).
            text: Text chunk to synthesize (pre-chunked by orchestrator, non-empty).
            voice: Voice identifier (provider-specific string).

        Contract:
        - Must emit exactly ONE terminal event per chunk:
            - TTSChunkComplete(run_id, chunk_index, pcm_bytes)
            OR
            - TTSError(run_id, reason)
        - pcm_bytes must be PCM16 16kHz mono format (conversion is adapter's job).
        - If run_id is not currently active, the implementation SHOULD ignore the
          request (no-op) rather than buffering it.
        - The adapter MUST NOT retry internally.
        - The adapter MUST NOT block the event loop indefinitely.
        """
        raise NotImplementedError

    @abstractmethod
    async def cancel(self, run_id: int) -> None:
        """
        Request cancellation of the specified TTS run.

        Contract:
        - After cancellation, the adapter must stop emitting events for that run_id
          as quickly as possible.
        - cancel() MUST be idempotent: repeated calls for the same run_id are safe.
        - If run_id is unknown or already complete, cancel() should be a no-op.
        - Adapter MAY emit TTSError(run_id, reason="cancelled") OR silently stop
          producing events.

        Notes:
        - The cancellation acknowledgment protocol (CANCEL_ACK / timeout) is handled
          outside the adapter by the runtime/cancellation layer.
        - Cancellation semantics are finalized by the orchestrator, not the adapter.
        """
        raise NotImplementedError
