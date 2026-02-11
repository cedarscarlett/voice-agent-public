"""
Service enumeration for run-id–versioned external streams.

Rules:
- This enum identifies versioned external services only.
- It must NOT encode behavior or lifecycle rules.
- Reducer logic decides how services are started, canceled, and reset.
"""

from __future__ import annotations

from enum import Enum


class Service(str, Enum):
    """
    External, versioned services managed by the orchestrator.

    Each service:
    - Has at most one active run at a time
    - Is identified by a monotonically increasing run_id
    """

    ASR = "ASR"
    LLM = "LLM"
    TTS = "TTS"
