"""
ASR adapter contract (Phase 6).

This module defines the *interface only* — no buffering, endpointing, retries,
timers, or orchestration decisions live here.

Key invariants (spec §6, §17, §18):
- Run IDs are owned by the orchestrator (monotonic per service). Adapters never
  generate or mutate run IDs.
- The adapter emits ASR events; it does not call the reducer or make state
  transitions.
- Cancellation is explicit: cancel(run_id) is a request to stop producing output
  for that run as quickly as possible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ASRAdapter(ABC):
    """
    Abstract interface for a streaming ASR adapter.

    Note: emit_event callback must be async.

    Implementations are responsible for:
    - Accepting PCM16 frames for a specific run_id via send_audio()
    - Producing ASR events (partials/final/endpoint/error) for that run_id
    - Supporting cancellation via cancel()

    Non-responsibilities:
    - No state machine logic (IDLE/LISTENING/etc.)
    - No endpointing policy decisions outside the adapter’s internal algorithm
    - No timers owned by the reducer
    - No direct interaction with WebSocket or UI
    """

    @abstractmethod
    async def start_stream(self, run_id: int) -> None:
        """
        Start a new ASR stream for the given run_id.

        The adapter will emit ASR events asynchronously via the configured
        event sink (callback or queue).
        """
        raise NotImplementedError

    @abstractmethod
    async def send_audio(self, run_id: int, sequence_num: int, pcm_bytes: bytes) -> None:
        """
        Provide one PCM16 audio frame to the ASR engine.

        Args:
            run_id: Orchestrator-owned run identifier for the current ASR stream.
            sequence_num: Monotonic frame sequence number (debug/observability).
            pcm_bytes: Raw PCM16 bytes for exactly one 20ms frame.

        Contract:
        - If run_id is not currently active, the implementation SHOULD ignore the
          frame (no-op) rather than buffering it for a future run.
        - The adapter MAY apply internal buffering, VAD, or endpoint detection,
          but must keep outputs deterministic for a given input stream.
        """
        raise NotImplementedError

    @abstractmethod
    async def cancel(self, run_id: int) -> None:
        """
        Request cancellation of the specified ASR run.

        Contract:
        - After cancellation, the adapter must stop emitting events for that run_id
          as quickly as possible.
        - cancel() MUST be idempotent: repeated calls for the same run_id are safe.
        - If run_id is unknown or already complete, cancel() should be a no-op.

        Notes:
        - The cancellation acknowledgment protocol (CANCEL_ACK / timeout) is handled
          outside the adapter by the runtime/cancellation layer.
        """
        raise NotImplementedError
