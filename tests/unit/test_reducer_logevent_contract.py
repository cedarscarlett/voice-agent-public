# pylint: disable=missing-module-docstring,missing-function-docstring

from orchestrator.reducer import reduce
from orchestrator.state_dataclass import OrchestratorState
from orchestrator.events import MicStart, EventType
from orchestrator.commands import LogEvent
from orchestrator.enums.state import State


def test_reducer_emits_logevent_with_required_fields():
    state = OrchestratorState(state=State.IDLE)

    event = MicStart(
        event_type=EventType.MIC_START,
        ts_ms=123,
    )

    _, commands = reduce(state, event)

    log_events = [c for c in commands if isinstance(c, LogEvent)]
    assert log_events, "Reducer must emit at least one LogEvent"

    payload = log_events[0].event

    # Spec §14 required fields
    assert "ts_ms" in payload
    assert "state" in payload
    assert "mode" in payload
    assert "event_type" in payload
    assert "decision" in payload
    assert "run_ids" in payload
    assert "queue_depths" in payload

    # Phase 4 placeholders (do not compute yet)
    assert payload["queue_depths"]["audio_in"] == 0.0
    assert payload["queue_depths"]["audio_out"] == 0.0
