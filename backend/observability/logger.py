"""
JSONL event logger.

Spec §14:
- Write one JSON object per line
- Output to stdout (v1)
- No buffering, no batching
- No side effects beyond logging
"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping, Callable


# ------------------------------------------------------------------
# Explicit output sink (patchable in tests, swappable in later phases)
# ------------------------------------------------------------------

def _stdout_print(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()

_print: Callable[[str], None] = _stdout_print

def log_event(event: Mapping[str, Any]) -> None:
    """
    Write a single JSONL event to stdout.

    The caller is responsible for:
    - Supplying a fully-formed event dict
    - Including ts_ms, session_id, state, etc.

    This function:
    - Serializes to JSON
    - Writes exactly one line
    - Flushes immediately (no buffering)
    - Never raises
    """
    try:
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as e:
        # Last-resort fallback — logging must never crash the runtime
        fallback: dict[str, Any] = {
            "ts_ms": event.get("ts_ms"),
            "event_type": "LOGGER_SERIALIZATION_ERROR",
            "error": str(e),
            "original_event_repr": repr(event),
        }
        line = json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))

    _print(line)
