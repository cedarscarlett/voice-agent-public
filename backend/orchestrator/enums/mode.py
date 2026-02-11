"""
Operational mode enumeration.

Modes are orthogonal to control states:
- State answers: "What is the system doing?"
- Mode answers:  "Under what operating conditions?"
"""

from __future__ import annotations

from enum import Enum


class Mode(str, Enum):
    """
    Operational mode of the session.

    NORMAL:
        System is operating within expected latency and buffer bounds.

    DEGRADED:
        Backpressure thresholds exceeded.
        Audio correctness is prioritized over UI freshness.
    """

    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
