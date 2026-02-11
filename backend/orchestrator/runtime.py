"""
Runtime execution shell for a single voice session.

Responsibilities:
- Own orchestrator state
- Call pure reducer
- Execute commands with side effects (ASR, timers, retries)
- Drain inbound audio queue into ASR based on orchestrator state
- Schedule and cancel timers
- Convert timer expiry into events

Non-responsibilities (for now):
- LLM/TTS execution (later step)
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from orchestrator.reducer import reduce
from orchestrator.commands import (
    Command,
    StartTTS,
    CancelTTS,
    EnqueueAudioFrames,
    LogEvent,
    StartASR,
    StopASR,
    StartTimer,
    CancelTimer,
    ScheduleRetry,
    CommitTurn,
    StartLLM,
    CancelLLM,
    ExecuteTool,
)
from orchestrator.enums.state import State
from orchestrator.enums.service import Service
from orchestrator.events import (
    Event,
    EventType,
    ASRFinalTimeout,
    LLMFirstTokenTimeout,
    LLMStallTimeout,
    TTSFirstAudioTimeout,
    TTSStallTimeout,
    ErrorTimeout,
    CancelTimeout,
    RetryReady,
    CancelAck,
    ToolResult,
)
from orchestrator.state_dataclass import OrchestratorState
from orchestrator.cancellation import CancellationManager

from observability.logger import log_event

from audio.frames import AudioFrame

from services.calendar_service import CalendarService


if TYPE_CHECKING:
    from orchestrator.runtime_context import RuntimeExecutionContext


def _now_ms() -> int:
    return time.time_ns() // 1_000_000

def should_send_to_ui(cmd: LogEvent) -> bool:
    """Decide if a LogEvent should be sent to the UI."""
    event = cmd.event
    event_type = event.get("event_type")

    return (
        event_type in {
            "START_ASR",
            "STOP_ASR",
            "ASR_PARTIAL",
            "ASR_FINAL",
            "START_LLM",
            "LLM_DONE",
            "CANCEL_LLM"
            "LLM_ERROR",
            "START_TTS",
            "CANCEL_TTS",
            "TTS_CHUNK_COMPLETE"
            "WS_CONNECTED",
            "WS_DISCONNECTED",
            "SESSION_STARTED",
            "SESSION_ENDED"
        }
        or event.get("decision") == "state_changed"
    )


class Runtime:
    """
    Runtime execution boundary for a single voice session.

    Responsibilities:
    - Own the authoritative orchestrator state
    - Act as the universal event sink for the session
      (gateway events, adapter events, timer events)
    - Invoke the pure reducer deterministically
    - Execute emitted commands with side effects
    - Schedule and cancel timers
    - Convert timer expiry into events

    Architectural role:
    Runtime is the bridge between the pure orchestration layer
    (reducer + immutable state) and the imperative world
    (adapters, logging, IO, time).

    Guarantees:
    - Reducer is always called exactly once per incoming event
    - State transitions are serialized and deterministic
    - All side effects occur *after* state has been updated
    - Runtime never performs orchestration logic itself
    - Timers emit events back into handle_event (single entry point)

    Non-responsibilities (by design, at this phase):
    - Owning adapters or queues
    - Transport concerns (WebSocket, binary protocols)
    - LLM / TTS execution (added in later steps)
    """

    def __init__(
        self,
        *,
        initial_state: OrchestratorState,
        context: RuntimeExecutionContext,
    ) -> None:
        self._tts_started: set[int] = set()
        self._state = initial_state
        self._ctx = context
        self._timers: dict[str, asyncio.Task[None]] = {}
        self._asr_first_frame_sent_for_run: int = -1
        self.calendar = CalendarService()

        # Runtime-owned monotonic TTS sequence counters (run_id -> next_seq)
        self._tts_seq: dict[int, int] = {}

        # Cancellation protocol infrastructure (no orchestration logic)
        self._cancellation = CancellationManager(
        emit_event=self.handle_event,
        hard_reset=self._hard_reset_service,
        get_active_run=self._active_run_for_service,
    )



    def _hard_reset_service(self, service: Service) -> None:
        """
        Perform a hard reset of a service adapter after cancel ACK timeout.

        This is a last-resort safety mechanism. No orchestration decisions
        are made here.

        TODO: Implement LLM using the same pattern.
        """
        if service is Service.ASR:
            if self._ctx.asr_adapter is not None:
                self._ctx.asr_adapter.force_reset()

        elif service is Service.TTS:
            if self._ctx.tts_adapter is not None:
                self._ctx.tts_adapter.force_reset()

        elif service is Service.LLM:
            if self._ctx.llm_adapter is not None:
                self._ctx.llm_adapter.force_reset()


    @property
    def state(self) -> OrchestratorState:
        """
        Return the current immutable orchestrator state.

        This state is the single source of truth for the session's
        control flow and lifecycle.

        Notes:
        - The returned object must be treated as read-only
        - State is only mutated internally by Runtime via the reducer
        - Gateway may mirror this value for observability only

        Consumers must never modify this state directly.
        """
        return self._state

    async def handle_event(self, event: Event) -> None:
        """
        Process a single event through the orchestration pipeline.

        Processing steps:
        1. Pass the current state and event to the pure reducer
        2. Atomically swap in the new orchestrator state
        3. Execute all emitted commands sequentially
        4. Drain audio if transitioning into LISTENING

        This method is the *only* entry point for events affecting
        orchestrator state. All event sources converge here:
        - Gateway (user input, connection lifecycle)
        - ASR adapter (partials, finals, errors)
        - Timers (timeouts, retry delays)

        Invariants:
        - State is updated before any side effects execute
        - Commands are executed in reducer-emitted order
        - No reducer logic is re-run during command execution
        - This method must be safe to call concurrently;
          internal serialization is enforced by the event loop

        This method intentionally returns nothing. All externally
        visible effects occur via side effects (adapters, logging).
        """
        # Cancellation ACK handling (infrastructure only)
        if isinstance(event, CancelAck):
            self._cancellation.notify_ack(
                service=event.service,
                run_id=event.run_id,
            )
        prev_state = self._state.state
        new_state, commands = reduce(self._state, event)
        if self._state.asr_authoritative != new_state.asr_authoritative:
            print(
                f"asr_authoritative: "
                f"{self._state.asr_authoritative} -> "
                f"{new_state.asr_authoritative}"
            )
        self._state = new_state

        for cmd in commands:
            await self._execute_command(cmd)

        # Drain backlog only on transition into LISTENING
        if (
            self._state.state is State.LISTENING
            and prev_state is not State.LISTENING
        ):
            await self._drain_audio_to_asr()

    async def notify_audio_enqueued(self) -> None:
        """
        Notification from gateway that audio has been enqueued.
        """
        log_event({
            "event_type": "mic_audio_seen",
            "state": self._state.state.value,
            "asr_run_id": self._state.active_runs.asr,
        })

        # Always drain audio; ASR adapter handles gating
        await self._drain_audio_to_asr()

    async def shutdown(self) -> None:
        """
        Clean shutdown of runtime.

        Cancels all in-flight timers and protocol managers and waits for tasks to complete.
        Called by gateway on session disconnect.
        """

        # Clear TTS sequence counters
        self._tts_seq.clear()

        for timer_id in list(self._timers.keys()):
            self._cancel_timer(timer_id)

        self._cancellation.clear_all()

        # Wait for all tasks to acknowledge cancellation
        if self._timers:
            await asyncio.gather(*self._timers.values(), return_exceptions=True)

    # ------------------------------------------------------------------
    # Command execution (side effects)
    # ------------------------------------------------------------------

    async def _execute_command(self, cmd: Command) -> None:
        """Execute a single command with side effects."""

        print("EXECUTING COMMAND:", type(cmd).__name__, getattr(cmd, "run_id", None))

        if isinstance(cmd, LogEvent):
            log_event({
                **cmd.event,
                "session_id": self._ctx.session_id,
                "connection_status": self._ctx.connection_status.value,
            })

        elif isinstance(cmd, StartASR):
            print("RUNTIME EXEC StartASR", {
                "t_ns": time.monotonic_ns(),
                "run_id": cmd.run_id,
            })
            assert self._ctx.asr_adapter is not None, "ASR adapter missing"
            await self._ctx.asr_adapter.start_stream(cmd.run_id)
            log_event({
                "ts_ms": _now_ms(),
                "event_type": "ASR_START_EXECUTED",
                "session_id": self._ctx.session_id,
                "asr_run_id": cmd.run_id,
            })

        elif isinstance(cmd, StopASR):
            assert self._ctx.asr_adapter is not None, "ASR adapter missing"
            await self._ctx.asr_adapter.cancel(cmd.run_id)


            # Start Cancel/ACK protocol
            self._cancellation.request_cancel(
                service=Service.ASR,
                run_id=cmd.run_id,
    )
            log_event({
                "ts_ms": _now_ms(),
                "event_type": "ASR_STOP_EXECUTED",
                "session_id": self._ctx.session_id,
                "asr_run_id": cmd.run_id,
            })

        elif isinstance(cmd, StartTimer):
            self._start_timer(
                timer_id=cmd.timer_id,
                duration_ms=cmd.duration_ms,
                timeout_event_type=cmd.timeout_event_type,
            )

        elif isinstance(cmd, CancelTimer):
            log_event({
                "event_type": "TIMER_CANCELLED",
                "timer_id": cmd.timer_id,
                "session_id": self._ctx.session_id,
            })
            self._cancel_timer(cmd.timer_id)

        elif isinstance(cmd, ScheduleRetry):
            self._schedule_retry(
                service=cmd.service,
                delay_ms=cmd.delay_ms,
                system_prompt_version=cmd.system_prompt_version,
                messages=cmd.messages,
            )

        elif isinstance(cmd, StartLLM):
            print("StartLLM received, messages len =", len(cmd.messages))
            assert self._ctx.llm_adapter is not None

            if not cmd.messages:
                print("llm_skipped_empty_messages")
                return

            await self._ctx.llm_adapter.start_completion(
                run_id=cmd.run_id,
                messages=list(cmd.messages),
            )

            log_event({
                "event_type": "llm_start_executed",
                "session_id": self._ctx.session_id,
                "llm_run_id": cmd.run_id,
                "system_prompt_version": cmd.system_prompt_version,
            })

        elif isinstance(cmd, CancelLLM):
            assert self._ctx.llm_adapter is not None
            await self._ctx.llm_adapter.cancel(cmd.run_id)

        # ------------------------------------------------------------
        # TTS commands
        # ------------------------------------------------------------

        elif isinstance(cmd, StartTTS):
            print("TTS START TEXT REPR:", repr(cmd.text))
            print("TTS START TEXT LEN:", len(cmd.text))
            assert self._ctx.tts_adapter is not None, "TTS adapter missing"

            # ------------------------------------------------------------
            # CONTROL PLANE: declare TTS run authoritative (ONCE per run)
            # ------------------------------------------------------------

            if cmd.run_id not in self._tts_started:
                self._tts_started.add(cmd.run_id)

                self._ctx.session.enqueue_control({
                    "type": "TTS_START",
                    "service": "TTS",
                    "run_id": cmd.run_id,
                    "ts_ms": _now_ms(),
                })

            # ------------------------------------------------------------
            # DATA PLANE: synthesize audio
            # ------------------------------------------------------------
            # Initialize sequence counter for this run if needed
            if cmd.run_id not in self._tts_seq:
                self._tts_seq[cmd.run_id] = 1

            await self._ctx.tts_adapter.synthesize_chunk(
                run_id=cmd.run_id,
                chunk_index=cmd.chunk_index,
                text=cmd.text,
                voice=cmd.voice,
            )

            log_event({
                "ts_ms": _now_ms(),
                "event_type": "tts_start_executed",
                "session_id": self._ctx.session_id,
                "tts_run_id": cmd.run_id,
                "chunk_index": cmd.chunk_index,
            })

        elif isinstance(cmd, CancelTTS):
            assert self._ctx.tts_adapter is not None, "TTS adapter missing"

            # 1. Stop TTS generation immediately
            await self._ctx.tts_adapter.cancel(cmd.run_id)

            # 2. HARD CUT: drop all pending outbound audio
            q = self._ctx.session.audio_out_queue
            if q is not None:
                q.clear()

            # 3) Tell the browser to stop playback immediately
            # This is the only way to kill already-buffered audio.
            self._ctx.session.enqueue_control({
                "type": "AUDIO_STOP",
                "service": "TTS",
                "run_id": cmd.run_id,
                "ts_ms": _now_ms(),
            })

            self._cancellation.request_cancel(
                service=Service.TTS,
                run_id=cmd.run_id,
            )

            log_event({
                "ts_ms": _now_ms(),
                "event_type": "tts_cancel_executed",
                "session_id": self._ctx.session_id,
                "tts_run_id": cmd.run_id,
            })

        elif isinstance(cmd, EnqueueAudioFrames):
            q = self._ctx.session.audio_out_queue
            if q is None:
                return

            print(
                "ENQUEUE_FRAME",
                {
                    "run_id": cmd.run_id,
                    "count": len(cmd.frames),
                    "t_ns": time.monotonic_ns(),
                },
            )

            # Initialize sequence counter defensively
            if cmd.run_id not in self._tts_seq:
                self._tts_seq[cmd.run_id] = 1

            for pcm_bytes in cmd.frames:
                seq = self._tts_seq[cmd.run_id]
                self._tts_seq[cmd.run_id] += 1

                frame = AudioFrame(
                    sequence_num=seq,
                    pcm_bytes=pcm_bytes,
                    ts_ms=_now_ms(),
                    run_id=cmd.run_id,
                )

                q.enqueue(frame)

            log_event({
                "ts_ms": _now_ms(),
                "event_type": "tts_audio_enqueued",
                "session_id": self._ctx.session_id,
                "tts_run_id": cmd.run_id,
                "frames": len(cmd.frames),
                "queue_depth_s": q.depth_seconds(),
            })

        elif isinstance(cmd, CommitTurn):
            ctx = self._ctx.session.conversation_context

            # Ignore stale or duplicate commits
            if cmd.turn_id <= self._ctx.session.last_committed_turn_id:
                return

            ctx.add_user_turn(cmd.user_text, cmd.turn_id)
            ctx.add_assistant_turn(cmd.assistant_text, cmd.turn_id)
            self._ctx.session.last_committed_turn_id = cmd.turn_id

            log_event({
                "ts_ms": _now_ms(),
                "event_type": "turn_committed",
                "session_id": self._ctx.session_id,
                "turn_id": cmd.turn_id,
            })

        elif isinstance(cmd, ExecuteTool):
            try:
                if cmd.tool == "check_calendar":
                    result = self.calendar.check_calendar(**cmd.args)
                elif cmd.tool == "book_appointment":
                    result = self.calendar.book_appointment(**cmd.args)
                else:
                    result = {"error": "unknown_tool", "tool": cmd.tool}
            except Exception as exc:  # pylint: disable=broad-exception-caught
                result = {"error": str(exc), "type": type(exc).__name__}

            await self.handle_event(
                ToolResult(
                    event_type=EventType.TOOL_RESULT,
                    ts_ms=_now_ms(),
                    tool=cmd.tool,
                    result=result,
                )
            )

        else:
            # Phase incremental: Only ASR + timers implemented so far
            # LLM/TTS commands will be handled in later steps
            log_event({
                "ts_ms": _now_ms(),
                "event_type": "COMMAND_NOT_YET_IMPLEMENTED",
                "session_id": self._ctx.session_id,
                "command_type": type(cmd).__name__,
            })

    # ------------------------------------------------------------------
    # Audio draining
    # ------------------------------------------------------------------

    async def _drain_audio_to_asr(self) -> None:
        """
        Drain audio_in_queue into ASR continuously.

        Audio is always forwarded; ASR adapter is authoritative
        for run gating via asr_authoritative.
        """
        q = self._ctx.audio_in_queue
        a = self._ctx.asr_adapter
        if q is None or a is None:
            print("ASR drain returning because q or a is None")
            return

        while True:
            frame = q.dequeue()
            if frame is None:
                break

            await a.send_audio(
                run_id=self._state.active_runs.asr,
                sequence_num=frame.sequence_num,
                pcm_bytes=frame.pcm_bytes,
            )

            log_event({
                "ts_ms": _now_ms(),
                "event_type": "ASR_AUDIO_SENT",
                "session_id": self._ctx.session_id,
                "asr_run_id": self._state.active_runs.asr,
                "seq_num": frame.sequence_num,
                "queue_depth_s": q.depth_seconds(),
            })

    # ------------------------------------------------------------------
    # Timer management
    # ------------------------------------------------------------------

    def _start_timer(
        self,
        *,
        timer_id: str,
        duration_ms: int,
        timeout_event_type: EventType,
    ) -> None:
        """
        Start or replace a timer that emits a timeout event.

        Timer tasks re-enter handle_event() when they expire,
        maintaining the single event entry point invariant.

        Timers are the mechanism for all temporal behavior:
        - ASR final timeout
        - LLM first token timeout
        - LLM stall timeout
        - TTS first audio timeout
        - TTS stall timeout
        - Cancel acknowledgment timeout
        - Error state auto-resolve delay
        """
        # Cancel existing timer if present (idempotent)
        self._cancel_timer(timer_id)

        async def _timer_task() -> None:
            try:
                await asyncio.sleep(duration_ms / 1000.0)

                # Construct timeout event based on type
                event = self._construct_timeout_event(
                    timer_id=timer_id,
                    timeout_event_type=timeout_event_type,
                )

                # Re-enter runtime with timeout event
                await self.handle_event(event)

            except asyncio.CancelledError:
                # Timer was cancelled - this is normal
                return

        self._timers[timer_id] = asyncio.create_task(_timer_task())

    def _cancel_timer(self, timer_id: str) -> None:
        """
        Cancel an in-flight timer if it exists.

        Idempotent: safe to call even if timer doesn't exist.
        """
        task = self._timers.pop(timer_id, None)
        if task is not None and not task.done():
            task.cancel()

    def _schedule_retry(
        self,
        *,
        service: Service,
        delay_ms: int,
        system_prompt_version: str,
        messages: tuple[dict[str, Any], ...],
    ) -> None:
        """
        Schedule a retry attempt.

        This is semantic sugar over _start_timer that emits RetryReady.
        Retry is just a delayed event carrying context for the new attempt.
        """
        timer_id = f"retry:{service.value}"

        # Cancel any existing retry for this service
        self._cancel_timer(timer_id)

        async def _retry_task() -> None:
            try:
                await asyncio.sleep(delay_ms / 1000.0)

                event = RetryReady(
                    event_type=EventType.RETRY_READY,
                    ts_ms=_now_ms(),
                    service=service,
                    system_prompt_version=system_prompt_version,
                    messages=messages,
                )

                await self.handle_event(event)

            except asyncio.CancelledError:
                return

        self._timers[timer_id] = asyncio.create_task(_retry_task())

    def _construct_timeout_event(
        self,
        *,
        timer_id: str,
        timeout_event_type: EventType,
    ) -> Event:
        """
        Construct the appropriate timeout event based on type.

        This is where runtime context (run_ids, timestamps) gets injected.
        The reducer emits timer commands with just EventType; runtime
        constructs the full event with current state.
        """
        ts = _now_ms()

        if timeout_event_type is EventType.ASR_FINAL_TIMEOUT:
            return ASRFinalTimeout(
                event_type=EventType.ASR_FINAL_TIMEOUT,
                ts_ms=ts,
                service=Service.ASR,
                run_id=self._state.active_runs.asr,
            )

        elif timeout_event_type is EventType.LLM_FIRST_TOKEN_TIMEOUT:
            return LLMFirstTokenTimeout(
                event_type=EventType.LLM_FIRST_TOKEN_TIMEOUT,
                ts_ms=ts,
                run_id=self._state.active_runs.llm,
            )

        elif timeout_event_type is EventType.LLM_STALL_TIMEOUT:
            return LLMStallTimeout(
                event_type=EventType.LLM_STALL_TIMEOUT,
                ts_ms=ts,
                run_id=self._state.active_runs.llm,
            )

        elif timeout_event_type is EventType.TTS_FIRST_AUDIO_TIMEOUT:
            return TTSFirstAudioTimeout(
                event_type=EventType.TTS_FIRST_AUDIO_TIMEOUT,
                ts_ms=ts,
                run_id=self._state.active_runs.tts,
            )

        elif timeout_event_type is EventType.TTS_STALL_TIMEOUT:
            return TTSStallTimeout(
                event_type=EventType.TTS_STALL_TIMEOUT,
                ts_ms=ts,
                run_id=self._state.active_runs.tts,
            )

        elif timeout_event_type is EventType.ERROR_TIMEOUT:
            return ErrorTimeout(
                event_type=EventType.ERROR_TIMEOUT,
                ts_ms=ts,
            )

        elif timeout_event_type is EventType.CANCEL_TIMEOUT:
            # Timer ID format: "cancel_ack:SERVICE:RUN_ID"
            # This matches the format from reducer._timer_id_cancel_ack()
            parts = timer_id.split(":")
            if len(parts) == 3 and parts[0] == "cancel_ack":
                service = Service(parts[1])
                run_id = int(parts[2])

                return CancelTimeout(
                    event_type=EventType.CANCEL_TIMEOUT,
                    ts_ms=ts,
                    service=service,
                    run_id=run_id,
                )

            # Fallback: malformed timer_id
            raise ValueError(
                f"Malformed cancel_ack timer_id: {timer_id} "
                f"(expected format: cancel_ack:SERVICE:RUN_ID)"
            )

        # Fallback: unknown timeout type
        # This should never happen if reducer is correct
        raise ValueError(
            f"Unknown timeout event type: {timeout_event_type} "
            f"for timer_id: {timer_id}"
        )

    def _active_run_for_service(self, service: Service) -> int:
        if service is Service.ASR:
            return self._state.active_runs.asr
        if service is Service.LLM:
            return self._state.active_runs.llm
        if service is Service.TTS:
            return self._state.active_runs.tts
