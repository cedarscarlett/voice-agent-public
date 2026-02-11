# pylint: disable=missing-module-docstring,missing-function-docstring

import json
from typing import Any

import pytest

import session.gateway as gateway_mod
from session.gateway import SessionGateway


class FakeASRAdapter:
    def start_stream(self, run_id: int) -> None:
        pass  # no-op for logging tests


def test_gateway_emits_logevent_via_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted: list[dict[str, Any]] = []

    def fake_log_event(payload: dict[str, Any]) -> None:
        emitted.append(payload)

    monkeypatch.setattr(gateway_mod, "log_event", fake_log_event)

    gw = SessionGateway()
    gw.on_ws_connect()

    # NEW: provision ASR adapter (required in Phase 6+)
    assert gw.session is not None
    gw.session.asr_adapter = FakeASRAdapter()

    gw.on_json_message(
        json.dumps(
            {
                "type": "MIC_START",
                "ts_ms": 123,
            }
        )
    )

    assert any(e["decision"] == "idle_to_listening" for e in emitted)
