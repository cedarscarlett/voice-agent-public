"""
Session gateway (Phase 6 — ASR Integration).

Responsibilities (spec §4, §12, §14, §15, §17):
- Owns VoiceSession lifecycle
- Tracks connection_status independently of orchestrator state
- Routes inbound JSON control messages -> orchestrator events
- Routes inbound binary audio frames -> ingest queue
- Executes ASR-related commands emitted by reducer (Phase 6)
- Detects sequence gaps and logs them
- Forwards events into runtime
- Executes ASR-related commands emitted by runtime
- Mirrors runtime state for observability only

Still NOT responsible for:
- Executing commands
- ASR / LLM / TTS adapters
- LLM or TTS command execution
- ASR policy decisions (endpointing, silence, timeouts)
- Any state machine logic
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from uuid import uuid4


from orchestrator.enums.mode import Mode
from orchestrator.events import (
    BargeIn,
    Event,
    EventType,
    MicStart,
    MicStop,
    SessionEnd,
    Stop,
    UserRetry,
    WSConnected,
    WSDisconnected,
    SessionStarted,
    SessionEnded
)
from orchestrator.runtime_context import RuntimeExecutionContext
from orchestrator.runtime import Runtime
from orchestrator.state_dataclass import OrchestratorState

from session.connection_status import ConnectionStatus
from session.voice_session import VoiceSession

from audio.queues import AudioFrameQueue
from protocol.binary import (
    decode_c2s_frame,
    check_sequence_gap,
    encode_s2c_frame,
    BinaryProtocolError,
)

from spec import (
    INGEST_AUDIO_Q_MAX_S,
    TTS_AUDIO_Q_MAX_S,
)

from observability.logger import log_event


from adapters.llm.streaming import StreamingLLMAdapter
from adapters.tts.speechmatics import SpeechmaticsTTSAdapter
from adapters.tts.elevenlabs_adapter import ElevenLabsTTSAdapter
from adapters.asr.deepgram_streaming import DeepgramStreamingASRAdapter

if TYPE_CHECKING:
    from config import AppConfig

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _now_ms() -> int:
    """Wall-clock milliseconds (fine for Phase 5)."""
    return time.time_ns() // 1_000_000


def _new_session_id() -> str:
    return f"sess_{uuid4().hex[:12]}"


# ------------------------------------------------------------------
# Gateway result
# ------------------------------------------------------------------

@dataclass(frozen=True)
class GatewayResult:
    """
    Return value for gateway boundary methods.

    outbound_json:
        JSON messages to send to client

    outbound_binary:
        Binary frames to send to client (audio)
    """
    outbound_json: tuple[dict[str, Any], ...] = ()
    outbound_binary: tuple[bytes, ...] = ()


# ------------------------------------------------------------------
# SessionGateway
# ------------------------------------------------------------------

class SessionGateway:
    """
    One gateway == one voice session.

    Phase 7: Now wires LLM adapter for end-to-end completion.
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        openai_client: Any | None = None,  # Type: openai.AsyncOpenAI
    ) -> None:
        self._config = config
        self.session: VoiceSession | None = None
        self._last_ingest_seq: int | None = None

        # LLM configuration (injected dependency, not global)
        self._openai_client = openai_client

    async def on_ws_connect(self) -> GatewayResult:
        """Called when a WebSocket connection is established."""
        session_id = _new_session_id()

        # Create session (context initializes in __post_init__)
        self.session = VoiceSession(session_id=session_id)
        self.session.connection_status = ConnectionStatus.UP

        # Initialize audio queues
        self.session.audio_in_queue = AudioFrameQueue(max_depth_s=INGEST_AUDIO_Q_MAX_S)
        self.session.audio_out_queue = AudioFrameQueue(max_depth_s=TTS_AUDIO_Q_MAX_S)

        # Resolve TTS voice (provider-agnostic)
        if self._config.tts_provider == "speechmatics":
            resolved_voice = self._config.speechmatics_voice or "sarah"
        else:
            resolved_voice = "default"  # or raise if unsupported

        # Create runtime (needs session for context)
        runtime = Runtime(
            initial_state=OrchestratorState(
                tts_voice=resolved_voice,
                conversation_context=self.session.conversation_context
            ),
            context=RuntimeExecutionContext(session=self.session),
        )

        # Construct TTS adapter
        if self._config.tts_provider == "speechmatics":
            assert self._config.speechmatics_api_key is not None
            tts_adapter = SpeechmaticsTTSAdapter(
                emit_event=runtime.handle_event,
                api_key=self._config.speechmatics_api_key,
                session_id=session_id,
            )
            self.session.attach_tts_adapter(tts_adapter)
        elif self._config.tts_provider == "elevenlabs":
            assert self._config.elevenlabs_api_key is not None
            assert self._config.elevenlabs_voice_id is not None
            assert self._config.elevenlabs_model_id is not None

            tts_adapter = ElevenLabsTTSAdapter(
                emit_event=runtime.handle_event,
                api_key=self._config.elevenlabs_api_key,
                session_id=session_id,
                voice_id=self._config.elevenlabs_voice_id,
                model_id=self._config.elevenlabs_model_id,
            )
            self.session.attach_tts_adapter(tts_adapter)
        else:
            raise RuntimeError(f"Unknown TTS_PROVIDER: {self._config.tts_provider}")

        # Construct LLM adapter
        # Attach LLM adapter ONLY if provided
        if self._openai_client is not None:
            llm_adapter = StreamingLLMAdapter(
                emit_event=runtime.handle_event,
                client=self._openai_client,
                model=self._config.llm_model,
                session_id=session_id,
                provider=self._config.llm_provider,
            )
            self.session.attach_llm_adapter(llm_adapter)

        # Construct ASR adapter
        assert self._config.deepgram_api_key is not None, "DEEPGRAM_API_KEY missing"
        asr_adapter = DeepgramStreamingASRAdapter(
            emit_event=runtime.handle_event,
            get_state=lambda: runtime.state,
            api_key=self._config.deepgram_api_key,
            model="flux-general-en",
            language=None,
            punctuate=True,
            smart_format=False,
            interim_results=True,
            vad_events=True,
        )
        self.session.attach_asr_adapter(asr_adapter)

        # Attach runtime (must be AFTER adapters)
        self.session.attach_runtime(runtime)

        self._last_ingest_seq = None

        await self._dispatch(
            SessionStarted(
                event_type=EventType.SESSION_STARTED,
                ts_ms=_now_ms(),
                session_id=session_id,
            )
        )

        # Dispatch WS_CONNECTED
        await self._dispatch(
            WSConnected(
                event_type=EventType.WS_CONNECTED,
                ts_ms=_now_ms(),
                session_id=session_id,
            )
        )

        init_msg: dict[str, Any] = {
            "type": "SESSION_INIT",
            "session_id": session_id,
            "audio_format": {
                "sample_rate": 16000,
                "sample_width": 2,
                "channels": 1,
                "frame_duration_ms": 20,
            },
            "config": {},
        }

        return GatewayResult(outbound_json=(init_msg,) + self._drain_control_out())

    async def on_ws_disconnect(self, reason: str | None = None) -> GatewayResult:
        """Called when the WebSocket disconnects."""
        if self.session is None:
            log_event({ # Initialize ASR adapter with direct runtime wiring
                "ts_ms": _now_ms(),
                "event_type": "WS_DISCONNECT_WITHOUT_SESSION",
                "reason": reason,
            })
            return GatewayResult()

        session_id = self.session.session_id

        runtime = self.session.runtime
        if runtime is not None:
            await runtime.shutdown()

        await self._dispatch(
            SessionEnded(
                event_type=EventType.SESSION_ENDED,
                ts_ms=_now_ms(),
                session_id=session_id,
            )
        )

        await self._dispatch(
            WSDisconnected(
                event_type=EventType.WS_DISCONNECTED,
                ts_ms=_now_ms(),
                session_id=session_id,
                reason=reason,
            )
        )

        return GatewayResult(
            outbound_json=self._drain_control_out(),
            outbound_binary=self._drain_audio_out(),
        )

    # ------------------------------------------------------------------
    # Audio egress
    # ------------------------------------------------------------------

    def _drain_audio_out(self) -> tuple[bytes, ...]:
        """
        Drain all queued TTS audio frames and encode for client.

        Called opportunistically after inbound activity.

        Early returns are redundant but kept for clarity.
        """
        if self.session is None:
#            print("drain: no session, returning")
            return ()

        q = self.session.audio_out_queue
        if q is None:
#            print("drain: no queue, returning")
            return ()

        frames: list[bytes] = []

#        print("ENTERING TTS AUDIO OUT DRAIN")

        while True:
#            print("DEQUEUING A TTS FRAME")
            frame = q.dequeue()
            if frame is None:
                break

            frames.append(
                encode_s2c_frame(
                    sequence_num=frame.sequence_num,
                    pcm_bytes=frame.pcm_bytes,
                    run_id=frame.run_id,
                )
            )

        if frames:
            log_event({
                "ts_ms": _now_ms(),
                "event_type": "tts_audio_drained",
                "session_id": self.session.session_id,
                "frames": len(frames),
                "queue_depth_s": q.depth_seconds(),
            })

        print(
            "DRAIN_BATCH",
            {
                "frames": len(frames),
                "t_ns": time.monotonic_ns(),
            },
        )

        return tuple(frames)

    async def on_json_message(self, payload: str) -> GatewayResult:
        """Route inbound JSON to orchestrator events."""
        if self.session is None:
            log_event({
                "ts_ms": _now_ms(),
                "event_type": "MESSAGE_WITHOUT_SESSION",
                "payload_preview": payload[:100],
            })
            return GatewayResult()

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            log_event({
                "ts_ms": _now_ms(),
                "event_type": "JSON_DECODE_ERROR",
                "session_id": self.session.session_id,
                "error": str(e),
                "payload_preview": payload[:100],
            })
            return GatewayResult()

        msg_type = data.get("type")
        ts_ms = data.get("ts_ms", _now_ms())

        event: Event | None = None

        if msg_type == "MIC_START":
            event = MicStart(event_type=EventType.MIC_START, ts_ms=ts_ms)
        elif msg_type == "MIC_STOP":
            event = MicStop(event_type=EventType.MIC_STOP, ts_ms=ts_ms)
        elif msg_type == "STOP":
            event = Stop(event_type=EventType.STOP, ts_ms=ts_ms)
        elif msg_type == "BARGE_IN":
            event = BargeIn(event_type=EventType.BARGE_IN, ts_ms=ts_ms)
        elif msg_type == "USER_RETRY":
            event = UserRetry(event_type=EventType.USER_RETRY, ts_ms=ts_ms)
        elif msg_type == "SESSION_END":
            event = SessionEnd(event_type=EventType.SESSION_END, ts_ms=ts_ms)
        else:
            log_event({
                "ts_ms": _now_ms(),
                "event_type": "UNKNOWN_MESSAGE_TYPE",
                "msg_type": msg_type,
                "session_id": self.session.session_id,
            })
            return GatewayResult()

        await self._dispatch(event)
        return GatewayResult(
            outbound_json=self._drain_control_out(),
            outbound_binary=self._drain_audio_out(),
        )


    async def on_binary_message(self, payload: bytes) -> GatewayResult:
        """
        Handle inbound binary mic audio frames.

        Phase 6 behavior:
        - Decode + validate
        - Detect sequence gaps
        - Enqueue into ingest queue (DEGRADED-aware)
        - Drain to ASR if LISTENING
        """
        if self.session is None:
            log_event({
                "ts_ms": _now_ms(),
                "event_type": "BINARY_WITHOUT_SESSION",
                "payload_len": len(payload),
            })
            return GatewayResult()

        try:
            frame = decode_c2s_frame(payload, ts_ms=_now_ms())
        except BinaryProtocolError as e:
            log_event({
                "ts_ms": _now_ms(),
                "event_type": "BINARY_DECODE_ERROR",
                "session_id": self.session.session_id,
                "error": str(e),
                "payload_len": len(payload),
            })
            return GatewayResult()

        # Sequence gap detection
        gap_result = check_sequence_gap(
            last_seq=self._last_ingest_seq,
            current_seq=frame.sequence_num,
        )
        if gap_result.gap:
            log_event({
                "ts_ms": _now_ms(),
                "event_type": "SEQ_GAP_DETECTED",
                "session_id": self.session.session_id,
                "expected": gap_result.expected,
                "actual": gap_result.actual,
                "gap_size": gap_result.gap_size,
            })

        self._last_ingest_seq = frame.sequence_num

        runtime = self.session.runtime
        assert runtime is not None, "Runtime must exist before audio enqueue"
        degraded = runtime.state.mode == Mode.DEGRADED

        # Enqueue with observability
        enqueue_ok = self.session.audio_in_queue.enqueue(frame, degraded=degraded)

        log_event({
            "ts_ms": _now_ms(),
            "event_type": "AUDIO_FRAME_ENQUEUED" if enqueue_ok else "AUDIO_FRAME_DROPPED",
            "session_id": self.session.session_id,
            "seq_num": frame.sequence_num,
            "queue_depth_s": self.session.audio_in_queue.depth_seconds(),
            "degraded": degraded,
            "runtime state": runtime.state.state
        })

        if not enqueue_ok:
            print("gateway: enqueue not ok")
            return GatewayResult()

        await runtime.notify_audio_enqueued()

        return GatewayResult(
            outbound_json=self._drain_control_out(),
            outbound_binary=self._drain_audio_out(),
        )

    # ------------------------------------------------------------------
    # Reducer dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, event: Event) -> None:
        """
        Forward event into runtime.

        Phase 4a:
        - Gateway no longer executes commands
        - Runtime owns all orchestration
        """
        if self.session is None:
            log_event({
                "ts_ms": _now_ms(),
                "event_type": "DISPATCH_WITHOUT_SESSION",
                "dropped_event": event.event_type.value,
            })
            return

        runtime = self.session.runtime
        assert runtime is not None, "Runtime must exist before dispatch"

        # Runtime handles everything
        await runtime.handle_event(event)

    def _drain_control_out(self) -> tuple[dict[str, Any], ...]:
        if self.session is None:
            return ()
        return self.session.drain_control()
