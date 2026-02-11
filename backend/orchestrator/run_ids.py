"""
Run ID container for versioned external services.

Rules:
- Run IDs are monotonic integers.
- They are owned and incremented ONLY by the orchestrator reducer.
- This module defines structure, not behavior.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunIds:
    """
    Immutable container for active run IDs per service.

    Semantics:
    - A value of 0 means "no run has been started yet".
    - Once a run ID is incremented, it is never reused.
    """

    asr: int = 0
    llm: int = 0
    tts: int = 0
