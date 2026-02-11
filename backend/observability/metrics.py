"""
Metrics and timing helpers for observability (Phase 4).

Responsibilities:
- Measure durations using monotonic time (immune to clock changes)
- Emit metrics as JSONL events via observability.logger
- Never aggregate: one metric = one log event
- Provide safe APIs that prevent timer leaks

Design notes:
- Durations use monotonic time for correctness
- Event timestamps (ts_ms) use wall-clock time for human readability
- Prefer the `timed()` context manager to avoid leaked timers
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Any

from observability.logger import log_event


# -----------------------------------------------------------------------------
# Internal timer storage
# -----------------------------------------------------------------------------
# timer_id -> (metric_name, start_time_ns)
_active_timers: dict[str, tuple[str, int]] = {}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def start_timer(name: str) -> str:
    """
    Start a monotonic timer.

    Returns:
        timer_id (str): Opaque ID required to stop the timer later.

    IMPORTANT:
        Callers MUST call stop_timer() in a finally block
        unless using the `timed()` context manager.

    Prefer:
        with timed("metric_name", session_id=...):
            ...
    """
    timer_id = f"timer_{uuid.uuid4().hex[:12]}"
    _active_timers[timer_id] = (name, time.monotonic_ns())
    return timer_id


def stop_timer(
    timer_id: str,
    *,
    session_id: str | None = None,
    state: str | None = None,
    details: dict[str, Any] | None = None,
    queue_depths_s: dict[str, float] | None = None,
) -> int | None:
    """
    Stop a previously started timer and emit a metric event.

    Args:
        timer_id: ID returned by start_timer()
        session_id: Optional session identifier
        state: Optional orchestrator state name
        details: Optional structured metadata
        queue_depths_s: Optional queue depths in seconds
            (Phase 4 placeholder; may be {"ingest": 0.0, "tts": 0.0})

    Returns:
        duration_ms if the timer existed, else None
    """
    entry = _active_timers.pop(timer_id, None)
    if entry is None:
        return None

    name, start_ns = entry
    duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000

    log_event({
        # Wall-clock timestamp for log correlation / readability
        # Duration correctness relies on monotonic time above
        "ts_ms": int(time.time() * 1000),
        "event_type": "METRIC_TIMER",
        "metric": name,
        "value_ms": duration_ms,
        "session_id": session_id,
        "state": state,
        "queue_depths_s": queue_depths_s or {},
        "details": details or {},
    })

    return duration_ms


# -----------------------------------------------------------------------------
# Safe API: context manager
# -----------------------------------------------------------------------------

@contextmanager
def timed(
    name: str,
    *,
    session_id: str | None = None,
    state: str | None = None,
    details: dict[str, Any] | None = None,
    queue_depths_s: dict[str, float] | None = None,
):
    """
    Context manager for measuring durations safely.

    Guarantees:
    - Timer is ALWAYS stopped (no leaks)
    - Metric is emitted exactly once
    - Exceptions inside the block do NOT suppress timing

    Usage:
        with timed(
            "llm_first_token_latency",
            session_id=session.session_id,
        ):
            await llm_stream()

    This is the preferred API for all timing in Phase 4+.
    """
    timer_id = start_timer(name)
    try:
        yield
    finally:
        stop_timer(
            timer_id,
            session_id=session_id,
            state=state,
            details=details,
            queue_depths_s=queue_depths_s,
        )
