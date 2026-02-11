"""
Conversation context management.

Responsibilities:
- Store ordered user/assistant turns
- Enforce truncation rules:
  - Max 8 turns OR max 6,000 characters (whichever is hit first)
  - Drop oldest turns until constraints are satisfied
  - Allow a single oversized turn (with warning)
- Provide a serializable representation for LLM consumption

Non-responsibilities:
- No reducer logic
- No LLM formatting
- No orchestration decisions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from observability.logger import log_event
from spec import MAX_CONTEXT_CHARS, MAX_CONTEXT_TURNS


Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Turn:
    """Single conversation turn."""
    role: Role
    text: str
    turn_id: int


class ConversationContext:
    """
    Mutable conversation context owned by the orchestrator.

    This object is intentionally imperative:
    - Reducer decides *when* to add turns
    - This class decides *what to keep*

    Invariants:
    - Turns are stored in chronological order
    - turn_id is monotonic but not required to be contiguous
    """

    def __init__(self, session_id: str | None = None) -> None:
        self._session_id = session_id
        self._turns: list[Turn] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_user_turn(self, text: str, turn_id: int) -> None:
        """Add a user turn and enforce truncation rules."""
        self._turns.append(Turn(role="user", text=text, turn_id=turn_id))
        self._truncate()

    def add_assistant_turn(self, text: str, turn_id: int) -> None:
        """Add an assistant turn and enforce truncation rules."""
        self._turns.append(Turn(role="assistant", text=text, turn_id=turn_id))
        self._truncate()

    def serialize(self) -> list[dict[str, str]]:
        """
        Serialize turns into a role/content structure.

        Output format:
        [
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."},
        ]
        """
        return [
            {"role": t.role, "content": t.text}
            for t in self._turns
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _truncate(self) -> None:
        """
        Enforce context size limits.

        Rules:
        - Max 8 turns OR max 6,000 characters
        - Drop oldest turns until valid
        - Allow a single oversized turn (log warning)
        """
        while self._violates_limits():
            # If only one turn remains, allow it even if oversized
            if len(self._turns) == 1:
                log_event({
                    "event_type": "context_single_turn_oversized",
                    "session_id": self._session_id,
                    "turn_id": self._turns[0].turn_id,
                    "char_count": len(self._turns[0].text),
                })
                break

            dropped = self._turns.pop(0)
            log_event({
                "event_type": "context_turn_dropped",
                "session_id": self._session_id,
                "turn_id": dropped.turn_id,
                "role": dropped.role,
                "char_count": len(dropped.text),
            })

    def _violates_limits(self) -> bool:
        """Return True if turn or character limits are exceeded."""
        if len(self._turns) > MAX_CONTEXT_TURNS:
            return True

        total_chars = sum(len(t.text) for t in self._turns)
        return total_chars > MAX_CONTEXT_CHARS
