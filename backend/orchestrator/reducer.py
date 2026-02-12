# pylint: disable=too-many-lines
"""
Pure orchestrator reducer.

(state, event) -> (new_state, commands)

Rules:
- Pure: no side effects, no IO, no clocks.
- Deterministic: output depends only on inputs.
- Total: every (state, event) pair is handled or explicitly ignored (logged).
"""

#   CancelTimer(TIMER_TTS_STALL)
#   CancelTimer(TIMER_TTS_FIRST_AUDIO)
# Reducer owns timer semantics; runtime must not cancel timers implicitly.

from __future__ import annotations


from dataclasses import replace
from typing import Any

import hashlib
import time

import json

from orchestrator.commands import (
    StartTTS,
    EnqueueAudioFrames,
    CancelLLM,
    CancelTTS,
    CancelTimer,
    Command,
    EndSession,
    LogEvent,
    StartASR,
    StartLLM,
    StartTimer,
    StopASR,
    ScheduleRetry,
    CommitTurn,
    ExecuteTool
)
from orchestrator.enums.service import Service
from orchestrator.enums.state import State
from orchestrator.events import (
    TTSChunkComplete,
    ASRError,
    ASRFinal,
    ASRPartial,
    CancelAck,
    CancelTimeout,
    ChunkTimerTick,
    EndOfSpeech,
    ErrorTimeout,
    Event,
    EventType,
    FatalError,
    LLMError,
    LLMFirstTokenTimeout,
    LLMDone,
    LLMStallTimeout,
    LLMToken,
    MicStart,
    MicStop,
    SessionStarted,
    SessionEnded,
    SessionEnd,
    Stop,
    TTSError,
    TTSFirstAudioTimeout,
    TTSStallTimeout,
    UserRetry,
    WSDisconnected,
    WSConnected,
    RetryReady,
    UserSpeechDetected,
    TTSAudioFrame,
    LLMToolCall,
    ToolResult
)

from orchestrator.retry import (
    FailureType,
    get_retry_delay_ms,
    next_attempt,
    should_retry,
    reset_attempt
)

from orchestrator.chunking import evaluate_chunk
from orchestrator.run_ids import RunIds
from orchestrator.state_dataclass import ChunkTimerState, OrchestratorState
from spec import (
    ASR_FINAL_TIMEOUT_MS,
    ERROR_STATE_AUTO_RESOLVE_DELAY_MS,
    LLM_FIRST_TOKEN_TIMEOUT_MS,
    LLM_STALL_TIMEOUT_S,
    TTS_FIRST_AUDIO_TIMEOUT_MS,
    TTS_STALL_TIMEOUT_S,
    TTS_CHUNK_TIME_MS,
    TTS_SIZE_TRIGGER_CHARS,
    MAX_TTS_IN_FLIGHT
)

from adapters.llm.prompts import SYSTEM_PROMPT_V1, SYSTEM_PROMPT_VERSION

from context.serialization import serialize_for_llm


# =============================================================================
# Phase 2 invariants (Run IDs & Cancellation)
# =============================================================================
# - Run IDs are bumped ONLY on new start (ASR/LLM/TTS)
# - Cancellation never bumps run IDs
# - CancelAck clears cancel_in_flight for (service, run_id)
# - CancelTimeout implies adapter hard-reset and PROCEEDING anyway (spec §11)
# - No retry-on-cancel-failure logic exists

# =============================================================================
# Timer IDs
# =============================================================================

TIMER_ASR_FINAL = "asr_final_timeout"
TIMER_LLM_FIRST_TOKEN = "llm_first_token_timeout"
TIMER_LLM_STALL = "llm_stall_timeout"
TIMER_TTS_FIRST_AUDIO = "tts_first_audio_timeout"
TIMER_TTS_STALL = "tts_stall_timeout"
TIMER_ERROR_AUTO_RESOLVE = "error_auto_resolve"
TIMER_CANCEL_ACK_PREFIX = "cancel_ack"
TIMER_CHUNK_POLL = "chunk_timer_tick"


# =============================================================================
# Small helpers
# =============================================================================

def _maybe_finish_turn(
    state: OrchestratorState,
    event: Event,
) -> tuple[OrchestratorState, tuple[Command, ...]] | None:
    """
    Commit the assistant turn if both subsystems are done:

    - LLM finished generating text
    - No TTS synth operations remain in flight
    - No buffered text remains
    """
    if (
        state.llm_finished
        and state.tts_in_flight == 0
        and state.tts_text_buffer == ""
    ):
        turn_id = state.current_turn_id
        assistant_text = state.assistant_text_accum

        new_state = _reset_tts_turn_bundle(state)
        new_state = replace(
            new_state,
            state=State.LISTENING,
            current_turn_id=turn_id + 1,
            pending_user_turn_text="",
        )

        return (
            new_state,
            _logs_last((
                _log(
                    new_state,
                    event,
                    "state_changed",
                    {
                        "from_state": state.state.value,
                        "to_state": new_state.state.value,
                        "source": "turn_complete_streaming",
                    },
                ),
            ) + (
                CommitTurn(
                    turn_id=turn_id,
                    user_text=state.pending_user_turn_text,
                    assistant_text=assistant_text,
                ),
                _log(
                    new_state,
                    event,
                    "turn_complete_streaming",
                    {
                        "turn_id": turn_id,
                        "assistant_len": len(assistant_text),
                    },
                ),
            )),
        )

    return None

def _bump_run_id(active_runs: RunIds, service: Service) -> RunIds:
    if service is Service.ASR:
        return replace(active_runs, asr=active_runs.asr + 1)
    if service is Service.LLM:
        return replace(active_runs, llm=active_runs.llm + 1)
    if service is Service.TTS:
        return replace(active_runs, tts=active_runs.tts + 1)
    raise ValueError(service)


def _active_run_for(active_runs: RunIds, service: Service) -> int:
    if service is Service.ASR:
        return active_runs.asr
    if service is Service.LLM:
        return active_runs.llm
    if service is Service.TTS:
        return active_runs.tts
    raise ValueError(service)


def _timer_id_cancel_ack(service: Service, run_id: int) -> str:
    return f"{TIMER_CANCEL_ACK_PREFIX}:{service.value}:{run_id}"

def _ensure_tts_run_started(state: OrchestratorState) -> OrchestratorState:
    """
    Ensure TTS run_id is initialized for the current assistant turn.
    Bumps run_id exactly once (on first chunk).
    """
    if not state.tts_run_started_for_turn:
        new_runs = _bump_run_id(state.active_runs, Service.TTS)
        return replace(
            state,
            active_runs=new_runs,
            tts_run_started_for_turn=True,
        )
    return state

def _log(
    state: OrchestratorState,
    event: Event,
    decision: str,
    details: dict[str, Any] | None = None,
) -> LogEvent:
    return LogEvent(
        event={
            "ts_ms": event.ts_ms,
            "state": state.state.value,
            "mode": state.mode.value,
            "event_type": event.event_type.value,
            "decision": decision,
            "run_ids": {
                "asr": state.active_runs.asr,
                "llm": state.active_runs.llm,
                "tts": state.active_runs.tts,
            },
            "queue_depths": {
                "audio_out": 0.0,
            },
            "cancel_in_flight": [s.value for s in state.cancel_in_flight],
            "details": details or {},
        }
    )


def _logs_last(commands: tuple[Command, ...]) -> tuple[Command, ...]:
    non_logs: list[Command] = []
    logs: list[Command] = []
    state_change_logs: list[Command] = []

    for command in commands:
        if isinstance(command, LogEvent):
            if command.event.get("decision") == "state_changed":
                state_change_logs.append(command)
            else:
                logs.append(command)
        else:
            non_logs.append(command)

    return tuple(non_logs + logs + state_change_logs)


def _ignore(
    state: OrchestratorState, event: Event, reason: str
) -> tuple[OrchestratorState, tuple[Command, ...]]:
    return state, (_log(state, event, "ignore", {"reason": reason}),)


def _enter_error(
    state: OrchestratorState, event: Event, reason: str
) -> tuple[OrchestratorState, tuple[Command, ...]]:
    new_state = replace(state, state=State.ERROR, last_error=reason)
    return new_state, _logs_last((
        _log(
            new_state,
            event,
            "state_changed",
            {
                "from_state": state.state.value,
                "to_state": new_state.state.value,
                "source": "enter_error",
            },
        ),
        _log(new_state, event, "enter_error", {"reason": reason}),
        StartTimer(
            timer_id=TIMER_ERROR_AUTO_RESOLVE,
            duration_ms=ERROR_STATE_AUTO_RESOLVE_DELAY_MS,
            timeout_event_type=EventType.ERROR_TIMEOUT,
        ),
    ))


def _cancel_service(
    state: OrchestratorState,
    event: Event,
    service: Service,
    run_id: int,
) -> tuple[OrchestratorState, list[Command]]:
    """
    Phase 2 invariants:
    - Cancellation does NOT bump run IDs
    - Cancellation only requests stop + starts ACK timer
    - Completion is driven solely by CancelAck or CancelTimeout
    """
    cmds: list[Command] = []
    new_state = replace(
        state,
        cancel_in_flight=frozenset(set(state.cancel_in_flight) | {service}),
    )

    if service is Service.ASR:
        cmds.append(
            _log(
                new_state,
                event,
                "stop_asr",
                {
                    "asr_run_id": run_id,
                    "source": "cancel_service",
                },
            )
        )
        cmds.append(StopASR(run_id=run_id))
    elif service is Service.LLM:
        cmds.append(
            _log(
                new_state,
                event,
                "cancel_llm",
                {
                    "llm_run_id": run_id,
                    "source": "cancel_service",
                },
            )
        )
        cmds.append(CancelLLM(run_id=run_id))
        cmds.append(CancelTimer(timer_id=TIMER_LLM_FIRST_TOKEN))
        cmds.append(CancelTimer(timer_id=TIMER_LLM_STALL))
    elif service is Service.TTS:
        cmds.append(
            _log(
                new_state,
                event,
                "cancel_tts",
                {
                    "tts_run_id": run_id,
                    "source": "cancel_service",
                },
            )
        )
        cmds.append(CancelTTS(run_id=run_id))
        cmds.append(CancelTimer(timer_id=TIMER_TTS_FIRST_AUDIO))
        cmds.append(CancelTimer(timer_id=TIMER_TTS_STALL))


    cmds.append(
        _log(
            new_state,
            event,
            "request_cancel",
            {"service": service.value, "run_id": run_id},
        )
    )
    return new_state, list(_logs_last(tuple(cmds)))

# =============================================================================
# Streaming TTS turn-bundle reset
# =============================================================================

def _reset_tts_turn_bundle(state: OrchestratorState) -> OrchestratorState:
    """
    Reset all TTS/LLM->TTS streaming bookkeeping to a clean slate for a new turn
    or an aborted turn.

    IMPORTANT: This is a *state-only* helper (pure). Timer cancel commands are
    emitted by callers where appropriate.
    """
    return replace(
        state,
        # Streaming TTS
        tts_next_play_chunk_index=0,
        tts_ready_audio_chunks={},
        tts_run_started_for_turn=False,
        llm_finished=False,
        tts_in_flight=0,
        tts_chunk_index=0,
        assistant_text_accum="",
        pending_tts_chunks=(),

        # LLM→TTS chunking buffer + timer
        tts_text_buffer="",
        chunk_timer_state=ChunkTimerState(False, 0),

        # Legacy frozen-chunk fields (kept for safety until fully removed)
        tts_chunks=(),
        tts_current_chunk_index=0,
        tts_first_audio_received=False,
    )


# =============================================================================
# Reducer entrypoint
# =============================================================================

def reduce(
    state: OrchestratorState, event: Event
) -> tuple[OrchestratorState, tuple[Command, ...]]:
    """
    Pure reducer for the voice session state machine.

    Given the current orchestrator state and a single event, returns:
    - the next state
    - a tuple of commands describing required side effects

    Properties:
    - Deterministic: no IO, clocks, or randomness
    - Total: every (state, event) pair is handled or explicitly ignored
    - Version-safe: ignores events with stale run IDs
    """
    if isinstance(event, FatalError):
        fatal_state = replace(state, state=State.IDLE, last_error=event.reason)
        return (
            fatal_state,
            _logs_last((
                _log(
                    fatal_state,
                    event,
                    "state_changed",
                    {
                        "from_state": state.state.value,
                        "to_state": fatal_state.state.value,
                        "source": "fatal_error",
                    },
                ),
            ) + (
                _log(
                    state,
                    event,
                    "fatal_error",
                    {
                        "reason": event.reason,
                        "context": event.context or {}
                    }
                ),
                EndSession(reason=f"fatal:{event.reason}"),
            )),
        )

    if isinstance(event, SessionStarted):
        return state, (
            _log(state, event, "session_started", {"session_id": event.session_id}),
        )

    if isinstance(event, SessionEnded):
        return state, (
            _log(state, event, "session_ended", {"session_id": event.session_id}),
        )

    # ------------------------------------------------------------------
    # ERROR gating
    # ------------------------------------------------------------------
    if state.state is State.ERROR:
        if isinstance(event, (ErrorTimeout, UserRetry)):
            cmds: list[Command] = []

            # Cancel auto-resolve timer if user explicitly retries
            if isinstance(event, UserRetry):
                cmds.append(CancelTimer(timer_id=TIMER_ERROR_AUTO_RESOLVE))

            # Cancel any outstanding cancel-ack timers
            for service in state.cancel_in_flight:
                run_id = _active_run_for(state.active_runs, service)
                if run_id > 0:
                    cmds.append(
                        CancelTimer(
                            timer_id=_timer_id_cancel_ack(service, run_id)
                        )
                    )

            # --- Commit failed assistant turn ---
            turn_id = state.current_turn_id
            user_text = state.pending_user_turn_text

            assistant_text = (
                "Sorry — I couldn’t respond in time. Please try again."
                if isinstance(event, ErrorTimeout)
                else "Okay, let’s try again."
            )

            cmds.append(
                CommitTurn(
                    turn_id=turn_id,
                    user_text=user_text,
                    assistant_text=assistant_text,
                )
            )

            cmds.append(
                _log(
                    state,
                    event,
                    "error_resolved_commit",
                    {
                        "turn_id": turn_id,
                        "reason": state.last_error,
                        "assistant_len": len(assistant_text),
                    },
                )
            )

            # --- Transition back to LISTENING for next user turn ---
            prev_state = state
            new_state = _reset_tts_turn_bundle(state)
            new_state = replace(
                new_state,
                state=State.LISTENING,
                current_turn_id=turn_id + 1,
                pending_user_turn_text="",
                last_error=None,
                cancel_in_flight=frozenset(),
            )
            return (
                new_state,
                _logs_last(tuple(cmds) + (
                    _log(
                        new_state,
                        event,
                        "state_changed",
                        {
                            "from_state": prev_state.state.value,
                            "to_state": new_state.state.value,
                            "source": "error_resolved",
                        },
                    ),
                )),
            )

        return _ignore(state, event, "in_error_state")

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    if isinstance(event, WSConnected):
        return state, (_log(state, event, "ws_connected", {"session_id": event.session_id}),)

    if isinstance(event, WSDisconnected):
        cmds: list[Command] = []
        new_state = replace(state, asr_authoritative=False)

        for service in (Service.TTS, Service.LLM, Service.ASR):
            run_id = _active_run_for(new_state.active_runs, service)
            if run_id > 0:
                new_state, more = _cancel_service(new_state, event, service, run_id)
                cmds.extend(more)

        prev_state = new_state
        new_state = _reset_tts_turn_bundle(new_state)
        new_state = replace(new_state, state=State.IDLE)
        cmds.append(_log(state, event, "ws_disconnected", {"reason": event.reason}))
        return (
            new_state,
            _logs_last(tuple(cmds) + (
                _log(
                    new_state,
                    event,
                    "state_changed",
                    {
                        "from_state": prev_state.state.value,
                        "to_state": new_state.state.value,
                        "source": "ws_disconnected",
                    },
                ),
            )),
        )

    if isinstance(event, SessionEnd):
        new_state = replace(state, state=State.IDLE)
        return (
            new_state,
            _logs_last((
                _log(
                    new_state,
                    event,
                    "state_changed",
                    {
                        "from_state": state.state.value,
                        "to_state": new_state.state.value,
                        "source": "session_end",
                    },
                ),
                _log(state, event, "session_end"),
                EndSession(reason="client"),
            )),
        )

    # ------------------------------------------------------------------
    # Cancel protocol events
    # ------------------------------------------------------------------
    if isinstance(event, CancelAck):
        if event.run_id != _active_run_for(state.active_runs, event.service):
            return _ignore(state, event, "cancel_ack_stale")

        # Phase 2 invariant:
        # - ACK clears cancel_in_flight only
        # - No run_id bump
        # - No state transition
        new_state = replace(
            state,
            cancel_in_flight=frozenset(s for s in state.cancel_in_flight if s is not event.service),
        )
        return (
            new_state,
            (
                CancelTimer(timer_id=_timer_id_cancel_ack(event.service, event.run_id)),
                _log(
                    new_state,
                    event,
                    "cancel_ack",
                    {
                        "service": event.service.value,
                        "run_id": event.run_id
                    }
                ),
            ),
        )

    if isinstance(event, CancelTimeout):
        # Phase 2: cancellation ACK did not arrive.
        # Spec §11: hard-reset adapter, proceed anyway.
        # Runtime is responsible for force-killing the adapter.
        if event.run_id != _active_run_for(state.active_runs, event.service):
            return _ignore(state, event, "cancel_timeout_stale")
        new_state = replace(
            state,
            cancel_in_flight=frozenset(
                s for s in state.cancel_in_flight if s is not event.service
            ),
        )
        return (
            new_state,
            (
                _log(
                    new_state,
                    event,
                    "cancel_timeout_hard_reset",
                    {
                        "service": event.service.value,
                        "run_id": event.run_id,
                    },
                ),
            ),
        )

    # ------------------------------------------------------------------
    # State dispatch
    # ------------------------------------------------------------------

    if isinstance(event, UserSpeechDetected):
        # Case 1: user speaks while system is talking → barge-in
        if not state.asr_authoritative:
            cmds: list[Command] = []

            new_state = state
            for service in (Service.LLM, Service.TTS):
                run_id = _active_run_for(new_state.active_runs, service)
                if run_id > 0:
                    new_state, more = _cancel_service(new_state, event, service, run_id)
                    cmds.extend(more)

            new_runs = _bump_run_id(new_state.active_runs, Service.ASR)
            new_state = _reset_tts_turn_bundle(new_state)
            new_state = replace(
                new_state,
                asr_authoritative=True,
                state=State.LISTENING,
                active_runs=new_runs,
                asr_endpoint_ts_ms=None,
            )
            cmds.append(
                _log(
                    new_state,
                    event,
                    "start_asr",
                    {
                        "asr_run_id": new_runs.asr,
                        "source": "barge_in_restart",
                    },
                )
            )
            cmds.append(StartASR(run_id=new_runs.asr))
            cmds.append(_log(state, event, "barge_in"))
            cmds.append(
                _log(
                    new_state,
                    event,
                    "cancel TTS/LLM and restart ASR",
                    {"asr_run_id": new_runs.asr}
                )
            )
            return (
                new_state,
                _logs_last(tuple(cmds) + (
                    _log(
                        new_state,
                        event,
                        "state_changed",
                        {
                            "from_state": state.state.value,
                            "to_state": new_state.state.value,
                            "source": "barge_in_to_listening",
                        },
                    ),
                )),
            )

        else:
            new_runs = _bump_run_id(state.active_runs, Service.ASR)
            new_state = replace(
                state,
                asr_authoritative=True,
                active_runs=new_runs,
            )

            return (
                new_state,
                _logs_last((
                    _log(
                        new_state,
                        event,
                        "start_asr",
                        {
                            "asr_run_id": new_runs.asr,
                            "source": "speech_detected",
                        },
                    ),
                    StartASR(run_id=new_runs.asr),
                )),
            )

    # ============================
    # IDLE
    # ============================
    if state.state is State.IDLE:
        if isinstance(event, MicStart):
            new_runs = _bump_run_id(state.active_runs, Service.ASR)
            new_state = replace(
                state,
                asr_authoritative=True,
                state=State.LISTENING,
                active_runs=new_runs,
                asr_endpoint_ts_ms=None,
            )

            cmds: list[Command] = [
                _log(
                    new_state,
                    event,
                    "start_asr",
                    {
                        "asr_run_id": new_runs.asr,
                        "source": "mic_start",
                    },
                ),
                StartASR(run_id=new_runs.asr),
            ]

            return (
                new_state,
                _logs_last(tuple(cmds) + (
                    _log(
                        new_state,
                        event,
                        "state_changed",
                        {
                            "from_state": state.state.value,
                            "to_state": new_state.state.value,
                            "source": "mic_start",
                        },
                    ),
                )),
            )

        if isinstance(event, MicStop):
            return state, (_log(state, event, "mic_stop_noop_in_idle"),)

        if isinstance(event, Stop):
            return state, (_log(state, event, "stop_noop_in_idle"),)

        return _ignore(state, event, "idle_unhandled")

    # ============================
    # LISTENING
    # ============================
    if state.state is State.LISTENING:


        if isinstance(event, ASRPartial):
            if event.run_id != state.active_runs.asr:
                return _ignore(state, event, "asr_partial_stale")
            return (
                state,
                (
                    _log(
                        state,
                        event,
                        "asr_partial",
                        {
                            "len": len(event.text),
                            "stable": event.is_stable
                        }
                    ),
                )
            )

        if isinstance(event, EndOfSpeech):
            if event.run_id != state.active_runs.asr:
                return _ignore(state, event, "end_of_speech_stale")
            return (
                replace(state, asr_endpoint_ts_ms=event.ts_ms),
                _logs_last((
                    _log(state, event, "end_of_speech"),
                    StartTimer(
                        timer_id=TIMER_ASR_FINAL,
                        duration_ms=ASR_FINAL_TIMEOUT_MS,
                        timeout_event_type=EventType.ASR_FINAL_TIMEOUT,
                    ),
                )),
            )

        if isinstance(event, ASRFinal):
            if event.run_id != state.active_runs.asr:
                return _ignore(state, event, "asr_final_stale")

            if not event.text.strip():
                new_state, cancel_cmds = _cancel_service(
                    state,
                    event,
                    Service.ASR,
                    event.run_id,
                )
                return (
                    replace(new_state, asr_authoritative=False),
                    _logs_last(tuple(cancel_cmds) + (
                        _log(
                            new_state,
                            event,
                            "asr_final",
                            {
                                "final_len": len(event.text),
                                "is_empty": True,
                            },
                        ),
                        _log(
                            new_state,
                            event,
                            "asr_final_empty_stop_asr",
                        ),
                    )),
                )

            # Serialize for LLM (without mutating context)
            messages = serialize_for_llm(
                system_prompt=SYSTEM_PROMPT_V1,
                context=state.conversation_context,
                user_text=event.text,
            )

            # Hash prompt
            full_prompt = "".join(m["content"] for m in messages)
            prompt_hash = hashlib.sha256(full_prompt.encode()).hexdigest()[:8]

            new_runs = _bump_run_id(state.active_runs, Service.LLM)
            new_state = _reset_tts_turn_bundle(state)
            new_state = replace(
                new_state,
                asr_authoritative=False,
                state=State.PROCESSING,
                active_runs=new_runs,
                asr_endpoint_ts_ms=None,
                llm_retry_attempt=reset_attempt(),
                llm_first_token_received=False,
                pending_user_turn_text=event.text,  # Store for later commit
            )

            return (
                new_state,
                _logs_last((
                    _log(
                        new_state,
                        event,
                        "state_changed",
                        {
                            "from_state": state.state.value,
                            "to_state": new_state.state.value,
                            "source": "asr_final_to_processing",
                        },
                    ),
                ) + (
                    _log(
                        new_state,
                        event,
                        "asr_final",
                        {
                            "final_len": len(event.text),
                            "is_empty": False,
                            "llm_run_id": new_runs.llm,
                        },
                    ),
                    _log(
                        new_state,
                        event,
                        "stop_asr",
                        {
                            "asr_run_id": event.run_id,
                            "source": "asr_final_to_processing",
                        },
                    ),
                    StopASR(run_id=event.run_id),
                    CancelTimer(timer_id=TIMER_ASR_FINAL),
                    _log(
                        new_state,
                        event,
                        "listening_to_processing",
                        {
                            "final_len": len(event.text),
                            "turn_id": state.current_turn_id,
                            "llm_run_id": new_runs.llm,
                            "system_prompt_version": SYSTEM_PROMPT_VERSION,
                            "prompt_hash": prompt_hash,
                        },
                    ),
                    _log(
                        new_state,
                        event,
                        "start_llm",
                        {
                            "llm_run_id": new_runs.llm,
                            "source": "asr_final_to_processing",
                            "system_prompt_version": SYSTEM_PROMPT_VERSION,
                            "prompt_hash": prompt_hash,
                        },
                    ),
                    StartLLM(
                        run_id=new_runs.llm,
                        system_prompt_version=SYSTEM_PROMPT_VERSION,
                        messages=tuple(messages),
                    ),
                    StartTimer(
                        timer_id=TIMER_LLM_FIRST_TOKEN,
                        duration_ms=LLM_FIRST_TOKEN_TIMEOUT_MS,
                        timeout_event_type=EventType.LLM_FIRST_TOKEN_TIMEOUT,
                    ),
                )),
            )

        if isinstance(event, ASRError):
            if event.run_id != state.active_runs.asr:
                return _ignore(state, event, "asr_error_stale")
            return _enter_error(state, event, f"asr_error:{event.reason}")

        if isinstance(event, Stop):
            cmds: list[Command] = []
            if state.active_runs.asr > 0:
                new_state, more = _cancel_service(state, event, Service.ASR, state.active_runs.asr)
                cmds.extend(more)
                cmds.append(_log(state, event, "stop_listening"))
                final_state = replace(new_state, asr_authoritative=False, state=State.IDLE)
                return (
                    final_state,
                    _logs_last(tuple(cmds) + (
                        _log(
                            final_state,
                            event,
                            "state_changed",
                            {
                                "from_state": state.state.value,
                                "to_state": final_state.state.value,
                                "source": "stop_from_listening",
                            },
                        ),
                    )),
                )
            cmds.append(_log(state, event, "stop_listening"))
            final_state = replace(state, state=State.IDLE)
            return (
                final_state,
                _logs_last(tuple(cmds) + (
                    _log(
                        final_state,
                        event,
                        "state_changed",
                        {
                            "from_state": state.state.value,
                            "to_state": final_state.state.value,
                            "source": "stop_from_listening",
                        },
                    ),
                )),
            )

        if isinstance(event, MicStop):
            return state, (_log(state, event, "mic_stop_ignored_in_listening"),)

        return _ignore(state, event, "listening_unhandled")

    # ============================
    # PROCESSING
    # ============================
    if state.state is State.PROCESSING:
        if isinstance(event, RetryReady):
            if event.service is not Service.LLM:
                return _ignore(state, event, "retry_ready_wrong_service")

            # Hash prompt (messages are carried by RetryReady)
            full_prompt = "".join(m["content"] for m in event.messages)
            prompt_hash = hashlib.sha256(full_prompt.encode()).hexdigest()[:8]

            # Start the retry attempt as a *new run* (monotonic run_id bump).
            new_runs = _bump_run_id(state.active_runs, Service.LLM)
            new_state = replace(
                state,
                active_runs=new_runs,
                llm_first_token_received=False,
            )

            return (
                new_state,
                _logs_last((
                    _log(
                        new_state,
                        event,
                        "retry_ready_start_llm",
                        {
                            "service": event.service.value,
                            "attempt": state.llm_retry_attempt.attempt,
                            "llm_run_id": new_runs.llm,
                            "system_prompt_version": event.system_prompt_version,
                            "prompt_hash": prompt_hash,
                        },
                    ),
                    _log(
                        new_state,
                        event,
                        "start_llm",
                        {
                            "llm_run_id": new_runs.llm,
                            "source": "retry_ready",
                            "system_prompt_version": event.system_prompt_version,
                            "prompt_hash": prompt_hash,
                        },
                    ),
                    StartLLM(
                        run_id=new_runs.llm,
                        system_prompt_version=event.system_prompt_version,
                        messages=event.messages,
                    ),
                    StartTimer(
                        timer_id=TIMER_LLM_FIRST_TOKEN,
                        duration_ms=LLM_FIRST_TOKEN_TIMEOUT_MS,
                        timeout_event_type=EventType.LLM_FIRST_TOKEN_TIMEOUT,
                    ),
                )),
            )
        if isinstance(event, LLMToken):
            if event.run_id != state.active_runs.llm:
                return _ignore(state, event, "llm_token_stale")

            is_first_token = not state.llm_first_token_received

            new_buffer = state.tts_text_buffer + event.delta

            # Detect buffer transition
            buffer_was_empty = state.tts_text_buffer == ""
            buffer_is_non_empty = new_buffer != ""

            # Start chunk timer on empty → non-empty transition
            if buffer_was_empty and buffer_is_non_empty:
                new_chunk_timer = ChunkTimerState(
                    active=True,
                    start_ts_ms=event.ts_ms,
                )
            else:
                # Timer continues running (or stays inactive)
                new_chunk_timer = state.chunk_timer_state

            new_state = replace(
                state,
                tts_text_buffer=new_buffer,
                llm_first_token_received=True,
                chunk_timer_state=new_chunk_timer,
            )

            commands: list[Command] = []

            # Cancel first-token timer ONCE
            if is_first_token:
                commands.append(CancelTimer(timer_id=TIMER_LLM_FIRST_TOKEN))

            # Restart stall timer on every token
            commands.append(
                StartTimer(
                    timer_id=TIMER_LLM_STALL,
                    duration_ms=int(LLM_STALL_TIMEOUT_S * 1000),
                    timeout_event_type=EventType.LLM_STALL_TIMEOUT,
                )
            )

            # Calculate elapsed time for chunk evaluation
            elapsed_ms = (
                event.ts_ms - new_state.chunk_timer_state.start_ts_ms
                if new_state.chunk_timer_state.active
                else 0
            )

            # --------------------------------------------------
            # Streaming TTS chunk evaluation (Step 5)
            # --------------------------------------------------
            decision = evaluate_chunk(
                buffer=new_state.tts_text_buffer,
                elapsed_ms=elapsed_ms,
            )

            if decision.send and new_state.tts_in_flight < MAX_TTS_IN_FLIGHT:
                send_text = decision.send_text or ""

                # Check if this is the first chunk (before state update)
                is_first_chunk = new_state.tts_in_flight == 0

                # Ensure TTS run_id exists (first chunk only)
                new_state = _ensure_tts_run_started(new_state)

                commands.append(
                    StartTTS(
                        run_id=new_state.active_runs.tts,
                        chunk_index=new_state.tts_chunk_index,
                        text=send_text,
                        voice=new_state.tts_voice,  # ← Use new_state
                    )
                )

                # Update reducer state
                new_state = replace(
                    new_state,
                    tts_in_flight=new_state.tts_in_flight + 1,
                    tts_chunk_index=new_state.tts_chunk_index + 1,
                    tts_text_buffer=decision.remainder or "",
                    assistant_text_accum=new_state.assistant_text_accum + send_text,
                )

                # Start first-audio timer on first chunk
                if is_first_chunk:  # ← Use saved flag
                    commands.append(
                        StartTimer(
                            timer_id=TIMER_TTS_FIRST_AUDIO,
                            duration_ms=TTS_FIRST_AUDIO_TIMEOUT_MS,
                            timeout_event_type=EventType.TTS_FIRST_AUDIO_TIMEOUT,
                        )
                    )

                # Reset chunk timer per spec §7
                if new_state.tts_text_buffer:
                    new_state = replace(
                        new_state,
                        chunk_timer_state=ChunkTimerState(
                            active=True,
                            start_ts_ms=event.ts_ms,
                        ),
                    )
                else:
                    new_state = replace(
                        new_state,
                        chunk_timer_state=ChunkTimerState(False, 0),
                    )

            elif decision.send:
                new_state = replace(
                    new_state,
                    pending_tts_chunks=new_state.pending_tts_chunks + (decision.send_text or "",),
                    tts_text_buffer=decision.remainder or "",
                    chunk_timer_state=ChunkTimerState(False, 0),  # Stop timer
                )

                commands.append(
                    _log(
                        new_state,
                        event,
                        "chunk_queued",
                        {
                            "chunk_len": len(decision.send_text or ""),
                            "pending_count": len(new_state.pending_tts_chunks),
                        },
                    )
                )

            else:
                # --------------------------------------------------
                # Spec §7: reset timer if B/C fired without sending
                # --------------------------------------------------
                size_triggered = len(new_state.tts_text_buffer) >= TTS_SIZE_TRIGGER_CHARS
                time_triggered = elapsed_ms >= TTS_CHUNK_TIME_MS

                if new_state.tts_text_buffer and (size_triggered or time_triggered):
                    new_state = replace(
                        new_state,
                        chunk_timer_state=ChunkTimerState(
                            active=True,
                            start_ts_ms=event.ts_ms,
                        ),
                    )

            commands.append(
                _log(
                    new_state,
                    event,
                    "llm_token",
                    {
                        "delta_len": len(event.delta),
                        "buf_len": len(new_state.tts_text_buffer),
                        "elapsed_ms": elapsed_ms,  # ← Add for debugging
                    },
                )
            )

            return new_state, _logs_last(tuple(commands))

        if isinstance(event, LLMDone):
            if event.run_id != state.active_runs.llm:
                return _ignore(state, event, "llm_done_stale")

            remainder = state.tts_text_buffer

            new_state = replace(
                state,
                llm_finished=True,
                tts_text_buffer="",
                chunk_timer_state=ChunkTimerState(False, 0),  # ← Stop timer
            )

            commands: list[Command] = [
                CancelTimer(timer_id=TIMER_LLM_STALL),
            ]

            # --------------------------------------------------
            # Flush remainder into pending queue (if any)
            # --------------------------------------------------
            if remainder.strip():
                new_state = replace(
                    new_state,
                    pending_tts_chunks=new_state.pending_tts_chunks + (remainder,),
                )

                commands.append(
                    _log(
                        new_state,
                        event,
                        "llm_done_flush_to_pending",
                        {"remainder_len": len(remainder)},
                    )
                )

            if new_state.pending_tts_chunks and new_state.tts_in_flight < MAX_TTS_IN_FLIGHT:
                send_text = new_state.pending_tts_chunks[0]
                remaining = new_state.pending_tts_chunks[1:]

                is_first_chunk = new_state.tts_in_flight == 0
                new_state = _ensure_tts_run_started(new_state)

                commands.append(
                    StartTTS(
                        run_id=new_state.active_runs.tts,
                        chunk_index=new_state.tts_chunk_index,
                        text=send_text,
                        voice=new_state.tts_voice,
                    )
                )

                new_state = replace(
                    new_state,
                    tts_in_flight=new_state.tts_in_flight + 1,
                    tts_chunk_index=new_state.tts_chunk_index + 1,
                    pending_tts_chunks=remaining,
                    assistant_text_accum=new_state.assistant_text_accum + send_text,
                )

                if is_first_chunk:
                    commands.append(
                        StartTimer(
                            timer_id=TIMER_TTS_FIRST_AUDIO,
                            duration_ms=TTS_FIRST_AUDIO_TIMEOUT_MS,
                            timeout_event_type=EventType.TTS_FIRST_AUDIO_TIMEOUT,
                        )
                    )

            commands.append(
                _log(
                    new_state,
                    event,
                    "llm_done_streaming",
                    {
                        "tts_in_flight": new_state.tts_in_flight,
                        "remainder_len": len(remainder),
                    },
                )
            )

            maybe = _maybe_finish_turn(new_state, event)
            if maybe:
                s2, c2 = maybe
                return s2, _logs_last(tuple(commands) + c2)

            return new_state, _logs_last(tuple(commands))

        if isinstance(event, LLMError):
            if event.run_id != state.active_runs.llm:
                return _ignore(state, event, "llm_error_stale")
            # Retry only for "init-like" failures (before first token).
            if not state.llm_first_token_received and should_retry(
                service=Service.LLM,
                failure=FailureType.INIT_ERROR,
                attempt=state.llm_retry_attempt,
            ):
                delay_ms = get_retry_delay_ms(service=Service.LLM, attempt=state.llm_retry_attempt)

                # Request cancel for the failed run (Cancel/ACK protocol stays intact).
                new_state, cancel_cmds = _cancel_service(
                    state, event, Service.LLM, event.run_id
                )
                new_state = replace(
                    new_state,
                    llm_retry_attempt=next_attempt(state.llm_retry_attempt),
                )

                return (
                    new_state,
                    _logs_last(tuple(cancel_cmds) + (
                        _log(
                            new_state,
                            event,
                            "llm_error",
                            {
                                "llm_run_id": event.run_id,
                                "reason": event.reason,
                                "retry_scheduled": True,
                                "attempt": state.llm_retry_attempt.attempt,
                                "delay_ms": delay_ms,
                            },
                        ),
                        _log(
                            new_state,
                            event,
                            "llm_init_error_schedule_retry",
                            {
                                "attempt": state.llm_retry_attempt.attempt,
                                "delay_ms": delay_ms,
                            },
                        ),
                        ScheduleRetry(
                            service=Service.LLM,
                            delay_ms=delay_ms,
                            system_prompt_version="v1",
                            messages=(),
                        ),
                    )),
                )

            # Mid-stream errors (or no retries left) -> ERROR.
            err_state, err_cmds = _enter_error(state, event, f"llm_error:{event.reason}")
            return err_state, _logs_last((
                _log(
                    state,
                    event,
                    "llm_error",
                    {
                        "llm_run_id": event.run_id,
                        "reason": event.reason,
                        "retry_scheduled": False,
                    },
                ),
            ) + err_cmds)

        if isinstance(event, LLMFirstTokenTimeout):
            if event.run_id != state.active_runs.llm:
                return _ignore(state, event, "llm_first_token_timeout_stale")
            # First-token timeout is an init-phase failure: allow init retry per policy.
            if should_retry(
                service=Service.LLM,
                failure=FailureType.FIRST_TOKEN_TIMEOUT,
                attempt=state.llm_retry_attempt
            ):
                delay_ms = get_retry_delay_ms(service=Service.LLM, attempt=state.llm_retry_attempt)
                new_state, cancel_cmds = _cancel_service(
                    state, event, Service.LLM, event.run_id
                )
                new_state = replace(
                    new_state,
                    llm_retry_attempt=next_attempt(state.llm_retry_attempt),
                    llm_first_token_received=False,
                )
                return (
                    new_state,
                    _logs_last(tuple(cancel_cmds) + (
                        _log(
                            new_state,
                            event,
                            "llm_first_token_timeout_schedule_retry",
                            {
                                "attempt": state.llm_retry_attempt.attempt,
                                "delay_ms": delay_ms,
                            },
                        ),
                        ScheduleRetry(
                            service=Service.LLM,
                            delay_ms=delay_ms,
                            system_prompt_version="v1",
                            messages=(),
                        ),
                    )),
                )

            # No retries left: cancel then ERROR -> IDLE auto-resolve.
            new_state, more = _cancel_service(state, event, Service.LLM, event.run_id)
            err_state, err_cmds = _enter_error(new_state, event, "llm_first_token_timeout")
            return err_state, _logs_last(tuple(more) + err_cmds)

        if isinstance(event, LLMStallTimeout):
            if event.run_id != state.active_runs.llm:
                return _ignore(state, event, "llm_stall_timeout_stale")
            new_state, more = _cancel_service(state, event, Service.LLM, event.run_id)
            err_state, err_cmds = _enter_error(new_state, event, "llm_stall")
            return err_state, _logs_last(tuple(more) + err_cmds)

        if isinstance(event, Stop):
            cmds: list[Command] = []

            new_state = state

            # Cancel any active services
            for service in (Service.LLM, Service.TTS):
                run_id = _active_run_for(new_state.active_runs, service)
                if run_id > 0:
                    new_state, more = _cancel_service(new_state, event, service, run_id)
                    cmds.extend(more)

            # Cancel all timers that could exist during generation
            cmds.append(CancelTimer(timer_id=TIMER_LLM_STALL))
            cmds.append(CancelTimer(timer_id=TIMER_LLM_FIRST_TOKEN))
            cmds.append(CancelTimer(timer_id=TIMER_TTS_STALL))
            cmds.append(CancelTimer(timer_id=TIMER_TTS_FIRST_AUDIO))
            cmds.append(_log(state, event, "stop"))

            # Reset reducer state
            new_state = replace(
                new_state,
                state=State.IDLE,
                tts_run_started_for_turn=False,
                tts_text_buffer="",
                llm_finished=False,
                tts_in_flight=0,
                tts_chunk_index=0,
                tts_chunks=(),
                tts_current_chunk_index=0,
                tts_first_audio_received=False,
                chunk_timer_state=ChunkTimerState(False, 0),
            )

            return (
                new_state,
                _logs_last(tuple(cmds) + (
                    _log(
                        new_state,
                        event,
                        "state_changed",
                        {
                            "from_state": state.state.value,
                            "to_state": new_state.state.value,
                            "source": "stop_from_processing",
                        },
                    ),
                )),
            )

        if isinstance(event, ChunkTimerTick):
            return _ignore(state, event, "chunk_timer_tick_unwired_phase1")

        if isinstance(event, TTSAudioFrame):
            if event.run_id != state.active_runs.tts:
                return _ignore(state, event, "tts_audio_frame_stale")

            print(
                "REDUCER_FRAME",
                {
                    "run_id": event.run_id,
                    "seq": event.sequence_num,
                    "t_ns": time.monotonic_ns(),
                },
            )

            cmds = [
                EnqueueAudioFrames(
                    run_id=event.run_id,
                    frames=(event.pcm_bytes,),  # Single frame
                )
            ]

            if not state.tts_first_audio_received:
                cmds.append(CancelTimer(timer_id=TIMER_TTS_FIRST_AUDIO))
                state = replace(state, tts_first_audio_received=True)

            cmds.append(
                StartTimer(
                    timer_id=TIMER_TTS_STALL,
                    duration_ms=int(TTS_STALL_TIMEOUT_S * 1000),
                    timeout_event_type=EventType.TTS_STALL_TIMEOUT,
                )
            )

            cmds.append(
                _log(state, event, "tts_audio_frame", {"sequence": event.sequence_num})
            )

            return state, _logs_last(tuple(cmds))

        if isinstance(event, TTSChunkComplete):
            if event.run_id != state.active_runs.tts:
                return _ignore(state, event, "tts_chunk_complete_stale")

            print(
                "REDUCER_COMPLETE_ENTER",
                {
                    "chunk_index": event.chunk_index,
                    "tts_in_flight_before": state.tts_in_flight,
                    "t_ns": time.monotonic_ns(),
                },
            )

            # Decrement in-flight counter
            assert state.tts_in_flight > 0, (
                f"TTSChunkComplete with tts_in_flight={state.tts_in_flight}, "
                f"run_id={event.run_id}, chunk_index={event.chunk_index}"
            )
            state = replace(state, tts_in_flight=state.tts_in_flight - 1)
            print(
                "REDUCER_COMPLETE_AFTER_DECR",
                {
                    "chunk_index": event.chunk_index,
                    "tts_in_flight": state.tts_in_flight,
                    "t_ns": time.monotonic_ns(),
                },
            )

            cmds: list[Command] = []

            # PRIORITY 1 - pending queue
            if state.pending_tts_chunks and state.tts_in_flight < MAX_TTS_IN_FLIGHT:
                send_text = state.pending_tts_chunks[0]
                remaining = state.pending_tts_chunks[1:]

                is_first_chunk = state.tts_in_flight == 0
                state = _ensure_tts_run_started(state)

                cmds.append(
                    StartTTS(
                        run_id=state.active_runs.tts,
                        chunk_index=state.tts_chunk_index,
                        text=send_text,
                        voice=state.tts_voice,
                    )
                )

                state = replace(
                    state,
                    tts_in_flight=state.tts_in_flight + 1,
                    tts_chunk_index=state.tts_chunk_index + 1,
                    pending_tts_chunks=remaining,
                    assistant_text_accum=state.assistant_text_accum + send_text,
                )

                if is_first_chunk:
                    cmds.append(
                        StartTimer(
                            timer_id=TIMER_TTS_FIRST_AUDIO,
                            duration_ms=TTS_FIRST_AUDIO_TIMEOUT_MS,
                            timeout_event_type=EventType.TTS_FIRST_AUDIO_TIMEOUT,
                        )
                    )

                cmds.append(
                    _log(
                        state,
                        event,
                        "pending_chunk_sent",
                        {
                            "chunk_len": len(send_text),
                            "remaining": len(remaining),
                        }
                    )
                )
            else:
                # --------------------------------------------------
                # Streaming scheduler: re-evaluate buffer if capacity freed
                # --------------------------------------------------
                if (
                    state.tts_text_buffer
                    and state.tts_in_flight < MAX_TTS_IN_FLIGHT
                ):
                    elapsed_ms = (
                        event.ts_ms - state.chunk_timer_state.start_ts_ms
                        if state.chunk_timer_state.active
                        else 0
                    )

                    decision = evaluate_chunk(
                        buffer=state.tts_text_buffer,
                        elapsed_ms=elapsed_ms,
                    )

                    if decision.send:
                        send_text = decision.send_text or ""
                        is_first_chunk = state.tts_in_flight == 0

                        state = _ensure_tts_run_started(state)

                        print(
                            "REDUCER_START_TTS",
                            {
                                "chunk_index": state.tts_chunk_index,
                                "tts_in_flight": state.tts_in_flight,
                                "t_ns": time.monotonic_ns(),
                            },
                        )

                        cmds.append(
                            StartTTS(
                                run_id=state.active_runs.tts,
                                chunk_index=state.tts_chunk_index,
                                text=send_text,
                                voice=state.tts_voice,
                            )
                        )

                        state = replace(
                            state,
                            tts_in_flight=state.tts_in_flight + 1,
                            tts_chunk_index=state.tts_chunk_index + 1,
                            tts_text_buffer=decision.remainder or "",
                            assistant_text_accum=state.assistant_text_accum + send_text,
                        )

                        if is_first_chunk:
                            cmds.append(
                                StartTimer(
                                    timer_id=TIMER_TTS_FIRST_AUDIO,
                                    duration_ms=TTS_FIRST_AUDIO_TIMEOUT_MS,
                                    timeout_event_type=EventType.TTS_FIRST_AUDIO_TIMEOUT,
                                )
                            )

                        # Reset chunk timer per spec §7
                        if state.tts_text_buffer:
                            state = replace(
                                state,
                                chunk_timer_state=ChunkTimerState(
                                    active=True,
                                    start_ts_ms=event.ts_ms,
                                ),
                            )
                        else:
                            state = replace(
                                state,
                                chunk_timer_state=ChunkTimerState(False, 0),
                            )

            cmds.append(
                _log(
                    state,
                    event,
                    "tts_chunk_complete",
                    {
                        "chunk_index": event.chunk_index,
                        "tts_in_flight": state.tts_in_flight,
                        "audio_streamed": True,
                    },
                )
            )

            # Try to finish turn
            maybe = _maybe_finish_turn(state, event)
            if maybe:
                s2, c2 = maybe
                return s2, _logs_last(tuple(cmds) + c2)

            return state, _logs_last(tuple(cmds))

        if isinstance(event, TTSError):
            if event.run_id != state.active_runs.tts:
                return _ignore(state, event, "tts_error_stale")
            err_state, err_cmds = _enter_error(state, event, f"tts_error:{event.reason}")
            return err_state, _logs_last((
                _log(
                    state,
                    event,
                    "tts_error",
                    {
                        "tts_run_id": event.run_id,
                        "reason": event.reason,
                    },
                ),
                CancelTimer(timer_id=TIMER_TTS_STALL),
                CancelTimer(timer_id=TIMER_TTS_FIRST_AUDIO),
            ) + err_cmds)

        if isinstance(event, TTSFirstAudioTimeout):
            if event.run_id != state.active_runs.tts:
                return _ignore(state, event, "tts_first_audio_timeout_stale")
            new_state, more = _cancel_service(state, event, Service.TTS, event.run_id)
            err_state, err_cmds = _enter_error(state, event, "tts_first_audio_timeout")
            return (
                err_state,
                _logs_last(
                    tuple(more)
                    + (CancelTimer(timer_id=TIMER_TTS_FIRST_AUDIO),)
                    + err_cmds
                ),
            )


        if isinstance(event, TTSStallTimeout):
            if event.run_id != state.active_runs.tts:
                return _ignore(state, event, "tts_stall_timeout_stale")
            new_state, more = _cancel_service(state, event, Service.TTS, event.run_id)
            err_state, err_cmds = _enter_error(state, event, "tts_stall")
            return err_state, _logs_last(
                tuple(more) + (CancelTimer(timer_id=TIMER_TTS_STALL),) + err_cmds
            )

        if isinstance(event, LLMToolCall):
            if event.run_id != state.active_runs.llm:
                return _ignore(state, event, "llm_tool_call_stale")

            return state, _logs_last((
                _log(
                    state,
                    event,
                    "llm_tool_call",
                    {"tool": event.tool, "args": event.args},
                ),
                ExecuteTool(tool=event.tool, args=event.args),
                # Optional but recommended: tool call ends LLM generation phase for this run
                CancelTimer(timer_id=TIMER_LLM_STALL),
            ))

        if isinstance(event, ToolResult):
            # Build continuation prompt: include history + the current user turn +
            # the assistant text already spoken + tool result.
            messages: list[dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT_V1},
                *state.conversation_context.serialize(),
                {"role": "user", "content": state.pending_user_turn_text},
            ]

            # Include what we've already spoken this turn (very important for continuity)
            if state.assistant_text_accum.strip():
                messages.append({"role": "assistant", "content": state.assistant_text_accum})

            messages.append({
                "role": "user",
                "content": f"<tool_result tool=\"{event.tool}\">{json.dumps(event.result)}</tool_result>",
            })

            new_runs = _bump_run_id(state.active_runs, Service.LLM)
            new_state = replace(
                state,
                active_runs=new_runs,
                llm_first_token_received=False,
                # Do NOT reset assistant_text_accum; we want to continue appending
            )

            return (
                new_state,
                _logs_last((
                    _log(
                        new_state,
                        event,
                        "tool_result_resume_llm",
                        {"tool": event.tool},
                    ),
                    StartLLM(
                        run_id=new_runs.llm,
                        system_prompt_version=SYSTEM_PROMPT_VERSION,
                        messages=tuple(messages),
                    ),
                    StartTimer(
                        timer_id=TIMER_LLM_FIRST_TOKEN,
                        duration_ms=LLM_FIRST_TOKEN_TIMEOUT_MS,
                        timeout_event_type=EventType.LLM_FIRST_TOKEN_TIMEOUT,
                    ),
                )),
            )

        return _ignore(state, event, "processing_unhandled")

    return _ignore(state, event, "unknown_state")
