"""
LLM adapter contract (v1).

Purpose:
- Define the interface for streaming LLM completions.
- Enforce run_id versioning at the boundary.
- Keep all orchestration, retries, timing, and cancellation semantics
  OUT of the adapter.

Rules:
- This file contains NO logic.
- No retries.
- No chunking.
- No side effects beyond emitting events.
- No knowledge of TTS, UI, or state machine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMAdapter(ABC):
    """
    Abstract base class for streaming LLM adapters.

    The adapter is a *dumb pipe*:
    input -> vendor -> token events.

    Orchestrator responsibilities (NOT here):
    - When to start
    - When to cancel
    - Retry policy
    - Timeouts
    - Context construction
    - What to do with tokens
    """

    @abstractmethod
    async def start_completion(
        self,
        *,
        run_id: int,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """
        Start a streaming LLM completion.

        Contract:
        - Must emit zero or more LLMTOKEN events.
        - Must emit at most ONE terminal event:
            - LLMDone(run_id)
            OR
            - LLMError(run_id, reason)
        - All emitted events MUST carry the provided run_id.
        - Must NOT retry internally.
        - Must NOT block the event loop indefinitely.
        - LLMTOKEN events must contain incremental deltas, not full text snapshots.

        Args:
            run_id:
                Monotonic run identifier assigned by the orchestrator.
            system_prompt:
                Fully resolved system prompt text (versioning handled upstream).
            messages:
                Serialized conversation context.
                This data is immutable from the adapter’s perspective.
        """
        raise NotImplementedError

    @abstractmethod
    async def cancel(self, run_id: int) -> None:
        """
        Request cancellation of an in-flight completion.

        Contract:
        - Best-effort, idempotent.
        - Must NOT raise if run_id is unknown or already completed.
        - Must result in no further LLMTOKEN events for that run_id.
        - Adapter MAY emit LLMError(run_id, reason="cancelled") OR
          silently stop producing events.

        Cancellation semantics are finalized by the orchestrator,
        not the adapter.
        """
        raise NotImplementedError
