"""
Persistent Deepgram streaming ASR adapter (Deepgram VAD is authoritative).

Core model (IMPORTANT):
- The Deepgram WebSocket connection is SESSION-scoped and stays open across turns.
- Audio is sent continuously (including during SPEAKING) so Deepgram VAD can detect barge-in.
- We do NOT gate transcript admission on run_id existence.
- We DO gate transcript admission on OrchestratorState.asr_authoritative (source of truth).
- OrchestratorState.active_runs.asr is monotonic and is bumped on StartASR; it is NOT reset to 0
  on StopASR. Therefore, "active run" is determined ONLY by asr_authoritative, not run_id > 0.

Event behavior:
- Deepgram VAD SpeechStarted (or equivalent) => emit USER_SPEECH_DETECTED ONLY when
  asr_authoritative is False (i.e., system is not already in an ASR turn).
- Deepgram transcripts:
    - If asr_authoritative is False: drop transcript events.
    - If asr_authoritative is True: emit ASR_PARTIAL / ASR_FINAL and END_OF_SPEECH.
      END_OF_SPEECH is emitted once per turn (per run_id).

Design constraints:
- Adapter must not call reducer directly.
- Adapter must not own orchestrator state transitions.
- Adapter must not know about WebSockets/session gateway beyond its own Deepgram socket.
- Adapter may consult a provided state getter for gating (read-only).
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, TYPE_CHECKING

from websockets.legacy.client import (
    connect as ws_connect,
    WebSocketClientProtocol,
)

from orchestrator.enums.service import Service
from orchestrator.events import EventType
from orchestrator.events import (
    ASRPartial,
    ASRFinal,
    EndOfSpeech,
    ASRError,
    CancelAck,
    Event,
    UserSpeechDetected
)

from spec import (
    AUDIO_SAMPLE_RATE_HZ,
    AUDIO_BYTES_PER_FRAME_PCM,
    SILENCE_DETECTION_MS,
)

if TYPE_CHECKING:
    # Your project already has this dataclass; we only need the shape for typing.
    from orchestrator.state_dataclass import OrchestratorState


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class _TurnRuntime:
    """
    Mutable turn-local bookkeeping (cleared/advanced on StartASR boundary).

    NOTE: Turn boundaries are defined by:
      - asr_authoritative flipping False -> True, and
      - active_runs.asr bumping to a new monotonic run_id.
    """
    run_id: int
    endpoint_emitted: bool
    last_partial_text: str


class DeepgramStreamingASRAdapter:
    """
    Persistent Deepgram Live WebSocket adapter using Deepgram VAD.

    Public interface matches your existing adapter:
    - start_stream(run_id): called on StartASR
    - send_audio(run_id, sequence_num, pcm_bytes): called continuously
    - cancel(run_id): called on StopASR
    - force_reset(): emergency reset

    Gating:
    - Transcript admission is gated by OrchestratorState.asr_authoritative (via get_state()).
    - VAD SpeechStarted emits USER_SPEECH_DETECTED only when NOT authoritative.

    Connection lifecycle:
    - Connects on first start_stream() (or first send_audio if you choose).
    - Stays connected across turns.
    - Reconnects lazily on failure (next start_stream/send_audio).
    """

    # Deepgram sometimes sends lots of interim messages; this avoids spamming identical partials.
    _MIN_PARTIAL_EMIT_INTERVAL_MS = 50

    def __init__(
        self,
        *,
        emit_event: Callable[[Event], Coroutine[Any, Any, None]],
        get_state: Callable[[], "OrchestratorState"],
        api_key: str,
        model: str = "flux-general-en",
        language: str | None = None,
        punctuate: bool = True,
        smart_format: bool = False,
        interim_results: bool = True,
        vad_events: bool = True,
        endpointing_ms: int | None = None,
        utterance_end_ms: int | None = None,
    ) -> None:
        self._emit_async = emit_event
        self._get_state = get_state
        self._api_key = api_key

        self._model = model
        self._language = language
        self._punctuate = punctuate
        self._smart_format = smart_format
        self._interim_results = interim_results
        self._vad_events = vad_events

        # Match your spec unless overridden
        self._endpointing_ms = (
            endpointing_ms if endpointing_ms is not None
            else int(SILENCE_DETECTION_MS)
        )
        self._utterance_end_ms = utterance_end_ms

        self._ws: WebSocketClientProtocol | None = None
        self._recv_task: asyncio.Task[None] | None = None

        self._lock = asyncio.Lock()

        self._turn: _TurnRuntime | None = None
        self._last_partial_emit_ts_ms: int = 0

        self._last_audio_send_ts: float = 0.0
        self._closing: bool = False

    # -------------------------------------------------------------------------
    # Public helpers
    # -------------------------------------------------------------------------

    def _emit(self, event: Event) -> None:
        # Emission is intentionally fire-and-forget.
        asyncio.create_task(self._emit_async(event))

    async def start_stream(self, run_id: int) -> None:
        """
        Called on StartASR.

        IMPORTANT:
        - Does NOT start/stop audio flow.
        - Does NOT reset/close the websocket.
        - Simply ensures connection exists, and resets turn-local bookkeeping.
        """
        async with self._lock:
            await self._ensure_connected_locked()

            # Turn boundary is defined by the run_id bump (monotonic) and authoritative gate.
            self._turn = _TurnRuntime(
                run_id=run_id,
                endpoint_emitted=False,
                last_partial_text="",
            )
            self._last_partial_emit_ts_ms = 0

    async def send_audio(
            self,
            run_id: int,
            sequence_num: int, # pylint: disable=unused-argument
            pcm_bytes: bytes) -> None:
        """
        Send PCM16LE mono frames to Deepgram.

        NOTE:
        - Audio is sent regardless of asr_authoritative.
        - run_id/sequence_num are accepted for interface symmetry; Deepgram doesn't need them.

        If the websocket is down:
        - emit ASR_ERROR (best effort)
        - drop frame
        """
        # Validate frame size early. If your pipeline allows variable sizes, remove this.
        if len(pcm_bytes) != AUDIO_BYTES_PER_FRAME_PCM:
            raise ValueError(
                f"Deepgram expected {AUDIO_BYTES_PER_FRAME_PCM} PCM bytes, "
                f"got {len(pcm_bytes)} (header not stripped?)"
            )

        # Avoid holding the lock while awaiting network send if possible.
        ws: WebSocketClientProtocol | None
        async with self._lock:
            if self._closing:
                print("send_audio exiting early", {
                    "t_ns": time.monotonic_ns(),
                    "run_id": run_id,
                })
                return
            await self._ensure_connected_locked()
            ws = self._ws

        if ws is None:
            # Should not happen due to ensure_connected, but keep defensive behavior.
            self._emit(
                ASRError(
                    event_type=EventType.ASR_ERROR,
                    ts_ms=_now_ms(),
                    service=Service.ASR,
                    run_id=run_id,
                    reason="deepgram_ws_missing",
                )
            )
            return

        try:
            await ws.send(pcm_bytes)
            async with self._lock:
                self._last_audio_send_ts = time.monotonic()
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Tear down socket; reconnect lazily later.
            async with self._lock:
                await self._drop_connection_locked()
            self._emit(
                ASRError(
                    event_type=EventType.ASR_ERROR,
                    ts_ms=_now_ms(),
                    service=Service.ASR,
                    run_id=run_id,
                    reason=f"deepgram_send_failed: {e!r}",
                )
            )

    async def cancel(self, run_id: int) -> None:
        """
        Called on StopASR.

        IMPORTANT:
        - Does NOT close websocket.
        - Returns adapter to VAD-only mode (i.e., transcripts will be dropped because
          orchestrator.asr_authoritative will be False).
        - Optionally requests Deepgram finalize to flush server-side utterance state.
        """
        # We do not trust run_id for "active vs inactive" since you keep it monotonic.
        # It's still used for CancelAck correlation only.
        # Flux does not require/expect Nova-style "Finalize" for this adapter contract.

        # Clear turn-local bookkeeping (optional; safest to clear to avoid stale endpoint flags).
        async with self._lock:
            self._turn = None
            self._last_partial_emit_ts_ms = 0

        self._emit(
            CancelAck(
                event_type=EventType.CANCEL_ACK,
                service=Service.ASR,
                run_id=run_id,
                ts_ms=_now_ms(),
            )
        )

    def force_reset(self) -> None:
        """
        Emergency hard reset without awaiting.

        Must:
        - Not await
        - Not acquire locks
        - Not emit events
        - Leave adapter in a clean idle state

        NOTE: This DOES drop the websocket. Next start_stream()/send_audio reconnects.
        """
        self._closing = True

        ws = self._ws
        self._ws = None

        rt = self._recv_task
        self._recv_task = None
        if rt is not None and not rt.done():
            rt.cancel()

        self._turn = None

        if ws is not None:
            try:
                # Close best-effort (can't await).
                asyncio.create_task(ws.close())
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        self._closing = False

    # -------------------------------------------------------------------------
    # Connection management
    # -------------------------------------------------------------------------

    def _build_url(self) -> str:
        # Flux uses /v2/listen and a different event model (TurnInfo).
        # Keep constructor signature intact, but only pass Flux-relevant params here.
        params: dict[str, str] = {
            "model": self._model,
            "encoding": "linear16",
            "sample_rate": str(AUDIO_SAMPLE_RATE_HZ),
            # Closest knob to "endpointing_ms": force end-of-turn after this much time.
            "eot_timeout_ms": str(int(self._endpointing_ms)),
        }

        qs = urllib.parse.urlencode(params)
        return f"wss://api.deepgram.com/v2/listen?{qs}"

    async def _ensure_connected_locked(self) -> None:
        if self._ws is not None:
            return

        url = self._build_url()
        headers = {"Authorization": f"Token {self._api_key}"}

        try:
            self._ws = await ws_connect(
                url,
                extra_headers=headers,
                max_size=2**22,
                ping_interval=None,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._ws = None
            self._emit(
                ASRError(
                    event_type=EventType.ASR_ERROR,
                    ts_ms=_now_ms(),
                    service=Service.ASR,
                    run_id=self._get_state().active_runs.asr,
                    reason=f"deepgram_connect_failed: {e!r}",
                )
            )
            return

        # Start receiver loop once per connection.
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _drop_connection_locked(self) -> None:
        ws = self._ws
        self._ws = None

        if self._recv_task is not None and not self._recv_task.done():
            self._recv_task.cancel()
        self._recv_task = None

        if ws is not None:
            try:
                await ws.close()
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    # -------------------------------------------------------------------------
    # Background loops
    # -------------------------------------------------------------------------

    async def _recv_loop(self) -> None:
        """
        Receives Deepgram messages and emits orchestrator events.

        RULES:
        - VAD SpeechStarted:
            - If asr_authoritative is False => emit USER_SPEECH_DETECTED (barge-in trigger)
            - If asr_authoritative is True  => ignore (already in an ASR turn)
        - Transcripts:
            - If asr_authoritative is False => drop
            - If asr_authoritative is True  => emit partial/final bound to state.active_runs.asr
        """
        ws: WebSocketClientProtocol | None
        async with self._lock:
            ws = self._ws

        if ws is None:
            return

        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"[DG] _recv_loop CRASHED: {e!r}")
                    continue

                await self._handle_message(data)
        except asyncio.CancelledError:
            return
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Connection died; drop it and surface error.
            async with self._lock:
                await self._drop_connection_locked()
            self._emit(
                ASRError(
                    event_type=EventType.ASR_ERROR,
                    ts_ms=_now_ms(),
                    service=Service.ASR,
                    run_id=self._get_state().active_runs.asr,
                    reason=f"deepgram_recv_failed: {e!r}",
                )
            )

    async def _handle_message(self, data: dict[str, Any]) -> None:
        msg_type = data.get("type")

        # Diagnostics
        st = self._get_state()
        print(f"[DG MSG] type={msg_type}, state={st.state.value}, asr_auth={st.asr_authoritative}")
        if msg_type == "TurnInfo":
            print(f"[DG FLUX] TurnInfo event={data.get('event')}")

        # Flux errors (don't drop connection yet - wait for error taxonomy)
        if msg_type == "Error":
            self._emit(
                ASRError(
                    event_type=EventType.ASR_ERROR,
                    ts_ms=_now_ms(),
                    service=Service.ASR,
                    run_id=self._get_state().active_runs.asr,
                    reason=f"deepgram_flux_error: {data.get('code')} {data.get('description')}",
                )
            )
            return

        if msg_type != "TurnInfo":
            return

        event = data.get("event")
        raw_transcript = data.get("transcript")
        transcript = raw_transcript.strip() if isinstance(raw_transcript, str) else ""

        # StartOfTurn = barge-in trigger
        if event == "StartOfTurn":
            print("DG StartOfTurn", {
                "t_ns": time.monotonic_ns(),
            })
            st = self._get_state()
            if not st.asr_authoritative:
                print("Emitting UserSpeechDetected", {
                    "t_ns": time.monotonic_ns(),
                })
                self._emit(
                    UserSpeechDetected(
                        event_type=EventType.USER_SPEECH_DETECTED,
                        ts_ms=_now_ms(),
                    )
                )
            return

        # Gate transcripts on asr_authoritative
        st = self._get_state()
        if not st.asr_authoritative:
            return

        run_id = st.active_runs.asr

        async with self._lock:
            print("_handle_message lock acquired", {
                "t_ns": time.monotonic_ns(),
            })
            # Align turn runtime
            if self._turn is None or self._turn.run_id != run_id:
                self._turn = _TurnRuntime(
                    run_id=run_id,
                    endpoint_emitted=False,
                    last_partial_text="",
                )
                self._last_partial_emit_ts_ms = 0

            # Block ALL events after endpoint (including duplicate EndOfTurn)
            if self._turn.endpoint_emitted and event == "Update":
                return

            now_ms = _now_ms()

            if event == "Update":
                print("DG Update", {
                    "t_ns": time.monotonic_ns(),
                    "chars": len(transcript),
                })
                if not transcript:
                    return

                if (now_ms - self._last_partial_emit_ts_ms) < self._MIN_PARTIAL_EMIT_INTERVAL_MS:
                    return
                self._last_partial_emit_ts_ms = now_ms
                self._turn.last_partial_text = transcript

                print("DG ASRPartial", {
                    "t_ns": time.monotonic_ns(),
                    "chars": len(transcript),
                })
                self._emit(
                    ASRPartial(
                        event_type=EventType.ASR_PARTIAL,
                        ts_ms=now_ms,
                        service=Service.ASR,
                        run_id=run_id,
                        text=transcript,
                        ts_range=None,
                        is_stable=False,
                    )
                )

            elif event == "EndOfTurn":
                print("DG EndOfTurn", {
                    "t_ns": time.monotonic_ns(),
                    "chars": len(transcript),
                })
                emit_eos = False

                if not self._turn.endpoint_emitted:
                    self._turn.endpoint_emitted = True
                    emit_eos = True

                if emit_eos:
                    print("Emitting EndOfSpeech", {
                        "t_ns": time.monotonic_ns(),
                        "chars": len(transcript),
                    })
                    self._emit(
                        EndOfSpeech(
                            event_type=EventType.END_OF_SPEECH,
                            ts_ms=now_ms,
                            service=Service.ASR,
                            run_id=run_id,
                        )
                    )

                if transcript:
                    print("Emitting ASRFinal", {
                        "t_ns": time.monotonic_ns(),
                        "chars": len(transcript),
                    })
                    self._emit(
                        ASRFinal(
                            event_type=EventType.ASR_FINAL,
                            ts_ms=now_ms,
                            service=Service.ASR,
                            run_id=run_id,
                            text=transcript,
                            ts_range=None,
                        )
                    )
        print("_handle_message lock released (no early return)", {
            "t_ns": time.monotonic_ns(),
            "chars": len(transcript),
        })
