"""
Authoritative session state enumeration.

Rules:
- This enum defines ONLY the control-plane states.
- No behavior, no helper methods, no side effects.
- Transitions are defined exclusively in the reducer.
"""

from __future__ import annotations

from enum import Enum


class State(str, Enum):
    """
    High-level deterministic control states for a single voice session.

    These states represent orchestration intent, NOT connection status
    and NOT adapter lifecycles.
    """

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"
