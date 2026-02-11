"""
Phase 2 cancellation semantics tests.

Reducer-only guarantees:
- Run ID correctness
- Cancel ACK safety
- Cancel timeout = hard reset + proceed (NOT ERROR)
"""

from orchestrator.enums.service import Service
from orchestrator.enums.state import State
from orchestrator.events import CancelAck, CancelTimeout, EventType
from orchestrator.reducer import reduce
from orchestrator.run_ids import RunIds
from orchestrator.state_dataclass import OrchestratorState
from orchestrator.commands import LogEvent


def test_late_cancel_ack_is_ignored():
    """
    CancelAck for a stale run_id must be ignored.
    """
    state = OrchestratorState(
        state=State.PROCESSING,
        active_runs=RunIds(llm=2),
        cancel_in_flight=frozenset({Service.LLM}),
    )

    event = CancelAck(
        event_type=EventType.CANCEL_ACK,
        ts_ms=0,
        service=Service.LLM,
        run_id=1,  # stale
    )

    new_state, cmds = reduce(state, event)

    assert new_state == state
    assert len(cmds) == 1
    assert isinstance(cmds[0], LogEvent)
    assert cmds[0].event["decision"] == "ignore"


def test_wrong_service_cancel_ack_is_safe():
    """
    CancelAck for a service not in cancel_in_flight is harmless.
    """
    state = OrchestratorState(
        state=State.PROCESSING,
        active_runs=RunIds(llm=1, tts=1),
        cancel_in_flight=frozenset({Service.TTS}),
    )

    event = CancelAck(
        event_type=EventType.CANCEL_ACK,
        ts_ms=0,
        service=Service.LLM,  # not in cancel_in_flight
        run_id=1,
    )

    new_state, cmds = reduce(state, event)

    # State unchanged
    assert new_state.state is State.PROCESSING
    assert new_state.cancel_in_flight == frozenset({Service.TTS})

    # Timer cleanup may still occur (idempotent)
    assert any(cmd.command_type.name == "CANCEL_TIMER" for cmd in cmds)


def test_cancel_timeout_hard_reset_proceeds():
    """
    CancelTimeout clears cancel_in_flight and does NOT enter ERROR.
    """
    state = OrchestratorState(
        state=State.PROCESSING,
        active_runs=RunIds(llm=3),
        cancel_in_flight=frozenset({Service.LLM}),
    )

    event = CancelTimeout(
        event_type=EventType.CANCEL_TIMEOUT,
        ts_ms=500,
        service=Service.LLM,
        run_id=3,
    )

    new_state, cmds = reduce(state, event)

    # No ERROR transition
    assert new_state.state is State.PROCESSING

    # cancel_in_flight cleared
    assert new_state.cancel_in_flight == frozenset()

    assert len(cmds) == 1
    assert isinstance(cmds[0], LogEvent)
    assert cmds[0].event["decision"] == "cancel_timeout_hard_reset"
