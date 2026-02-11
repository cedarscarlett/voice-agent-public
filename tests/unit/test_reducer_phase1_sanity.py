# pylint: disable=missing-module-docstring,missing-function-docstring
from orchestrator.reducer import reduce, TIMER_LLM_STALL
from orchestrator.state_dataclass import OrchestratorState, ChunkTimerState
from orchestrator.run_ids import RunIds
from orchestrator.enums.state import State
from orchestrator.enums.service import Service

from orchestrator.events import (
    EventType,
    MicStart,
    MicStop,
    Stop,
    BargeIn,
    UserRetry,
    ASRFinal,
    ASRPartial,
    LLMDone,
)

from orchestrator.commands import (
    Command,
    StartASR,
    CancelLLM,
    CancelTTS,
    CancelTimer,
    LogEvent,
)


# ---------------------------------------------------------------------
# Event helpers (mirror runtime construction)
# ---------------------------------------------------------------------

def mic_start(ts_ms: int = 0) -> MicStart:
    return MicStart(ts_ms=ts_ms, event_type=EventType.MIC_START)


def mic_stop(ts_ms: int = 0) -> MicStop:
    return MicStop(ts_ms=ts_ms, event_type=EventType.MIC_STOP)


def stop(ts_ms: int = 0) -> Stop:
    return Stop(ts_ms=ts_ms, event_type=EventType.STOP)


def barge_in(ts_ms: int = 0) -> BargeIn:
    return BargeIn(ts_ms=ts_ms, event_type=EventType.BARGE_IN)


def user_retry(ts_ms: int = 0) -> UserRetry:
    return UserRetry(ts_ms=ts_ms, event_type=EventType.USER_RETRY)


def asr_final(run_id: int, text: str = "hello") -> ASRFinal:
    return ASRFinal(
        ts_ms=0,
        event_type=EventType.ASR_FINAL,
        service=Service.ASR,
        run_id=run_id,
        text=text,
        ts_range=(0, 100),
    )


def asr_partial(
    run_id: int,
    text: str = "partial",
    stable: bool = False,
) -> ASRPartial:
    return ASRPartial(
        ts_ms=0,
        event_type=EventType.ASR_PARTIAL,
        service=Service.ASR,
        run_id=run_id,
        text=text,
        ts_range=(0, 50),
        is_stable=stable,
    )


def llm_done(run_id: int) -> LLMDone:
    return LLMDone(
        ts_ms=0,
        event_type=EventType.LLM_DONE,
        service=Service.LLM,
        run_id=run_id,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def decisions(commands: tuple[Command, ...]) -> list[str]:
    """Extract decision strings from LogEvent commands."""
    return [
        c.event["decision"]
        for c in commands
        if isinstance(c, LogEvent)
    ]


# ---------------------------------------------------------------------
# 1. Reducer shape & purity
# ---------------------------------------------------------------------

def test_reducer_returns_state_and_tuple():
    state = OrchestratorState()
    event = mic_start()

    new_state, commands = reduce(state, event)

    assert isinstance(commands, tuple)
    assert isinstance(new_state, OrchestratorState)


def test_reducer_does_not_mutate_input_state():
    state = OrchestratorState()
    event = mic_start()

    reduce(state, event)

    assert state.state is State.IDLE
    assert state.active_runs == RunIds()


# ---------------------------------------------------------------------
# 2. Run-ID gating (critical invariant)
# ---------------------------------------------------------------------

def test_stale_asr_final_is_ignored():
    state = OrchestratorState(
        state=State.LISTENING,
        active_runs=RunIds(asr=2),
    )

    event = asr_final(run_id=1)  # stale

    new_state, commands = reduce(state, event)

    assert new_state == state
    assert "ignore" in decisions(commands)


def test_stale_asr_partial_is_ignored():
    state = OrchestratorState(
        state=State.LISTENING,
        active_runs=RunIds(asr=5),
    )

    event = asr_partial(run_id=4)  # stale

    new_state, commands = reduce(state, event)

    assert new_state == state
    assert "ignore" in decisions(commands)


# ---------------------------------------------------------------------
# 3. Barge-in invariant
# ---------------------------------------------------------------------

def test_barge_in_cancels_llm_and_tts_and_restarts_asr():
    state = OrchestratorState(
        state=State.SPEAKING,
        active_runs=RunIds(asr=1, llm=2, tts=3),
        tts_text_buffer="hello world",
        chunk_timer_state=ChunkTimerState(active=True, elapsed_ms=200),
    )

    event = barge_in(ts_ms=1000)

    new_state, commands = reduce(state, event)

    assert new_state.state is State.LISTENING
    assert new_state.tts_text_buffer == ""
    assert new_state.chunk_timer_state.active is False
    assert new_state.active_runs.asr == 2  # bumped

    assert any(isinstance(c, StartASR) for c in commands)
    assert any(isinstance(c, CancelLLM) for c in commands)
    assert any(isinstance(c, CancelTTS) for c in commands)


# ---------------------------------------------------------------------
# 4. ERROR state gating & recovery
# ---------------------------------------------------------------------

def test_error_state_drops_events():
    state = OrchestratorState(
        state=State.ERROR,
        last_error="boom",
    )

    event = mic_start()

    new_state, commands = reduce(state, event)

    assert new_state.state is State.ERROR
    assert "ignore" in decisions(commands)


def test_user_retry_exits_error_and_cancels_timer():
    state = OrchestratorState(
        state=State.ERROR,
        last_error="boom",
    )

    event = user_retry()

    new_state, commands = reduce(state, event)

    assert new_state.state is State.IDLE
    assert new_state.last_error is None
    assert any(isinstance(c, CancelTimer) for c in commands)


# ---------------------------------------------------------------------
# 5. Timer cleanup on terminal events
# ---------------------------------------------------------------------

def test_llm_done_cancels_stall_timer():
    state = OrchestratorState(
        state=State.PROCESSING,
        active_runs=RunIds(llm=1),
    )

    event = llm_done(run_id=1)

    new_state, commands = reduce(state, event)

    assert new_state.state is State.IDLE
    assert any(
        isinstance(c, CancelTimer) and c.timer_id == TIMER_LLM_STALL
        for c in commands
    )


# ---------------------------------------------------------------------
# 6. Explicit no-op behavior (sanity)
# ---------------------------------------------------------------------

def test_stop_is_noop_in_idle():
    state = OrchestratorState(state=State.IDLE)
    event = stop()

    new_state, commands = reduce(state, event)

    assert new_state.state is State.IDLE
    assert "noop" in "".join(decisions(commands))


def test_mic_stop_is_ignored_in_listening():
    state = OrchestratorState(
        state=State.LISTENING,
        active_runs=RunIds(asr=1),
    )

    event = mic_stop()

    new_state, commands = reduce(state, event)

    assert new_state == state
    assert any("ignore" in d for d in decisions(commands))
