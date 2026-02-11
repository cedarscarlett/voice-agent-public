# pylint: disable=missing-module-docstring,missing-function-docstring

import json
from typing import Any

import pytest

from observability import logger


def test_log_event_emits_valid_jsonl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Phase 4 contract:
    - log_event emits exactly one JSONL line
    - payload is serialized as-is
    - output sink is patchable
    """
    captured: list[str] = []

    def fake_print(line: str) -> None:
        captured.append(line)

    # Patch the explicit output sink used by logger
    monkeypatch.setattr(logger, "_print", fake_print)

    payload: dict[str, Any] = {
        "event_type": "TEST",
        "value": 123,
    }

    logger.log_event(payload)

    # Exactly one line emitted
    assert len(captured) == 1

    # Must be valid JSON
    decoded = json.loads(captured[0])

    # Payload must be preserved exactly
    assert decoded == payload
