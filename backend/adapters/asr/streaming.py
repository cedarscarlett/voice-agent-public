"""
Deterministic streaming ASR adapter (Phase 6).

This adapter is the *only* ASR-layer component allowed to make ASR decisions:
- buffering
- silence detection / endpointing (spec §8)
- partial decode cadence + "stable" heuristics (spec §5)
- ASR_FINAL timeout handling (spec §5, §11) via emitting ASR_FINAL_TIMEOUT to reducer
- run-id gating
- cancellation + task cleanup
- event emission via injected sink (callback)

It must NOT:
- call the reducer directly
- own orchestration state machine transitions
- know about WebSockets, sessions, or queues

Design notes:
- Whisper is not truly streaming; partials are best-effort re-decodes of buffered audio.
- transcribe() is blocking; we run it in a thread to avoid stalling the event loop.
- Endpointing here is simple RMS-based silence detection over the last SILENCE_DETECTION_MS.
- If forced to choose, the system must prefer losing user audio
  over admitting assistant audio into ASR.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Optional, cast, Coroutine, Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from orchestrator.events import EventType
from orchestrator.enums.service import Service
from orchestrator.events import (
    ASRPartial,
    ASRFinal,
    EndOfSpeech,
    ASRError,
    Event,
    CancelAck
)
from spec import (
    AUDIO_SAMPLE_RATE_HZ,
    SILENCE_DETECTION_MS,
    AUDIO_BYTES_PER_FRAME_PCM
)

from adapters.asr.whisper_adapter import (
    WhisperEngine,
    WhisperBackendError,
    pcm16le_to_float32_mono,
    float32_to_pcm16le_mono,
)


if TYPE_CHECKING:
    from session.voice_session import VoiceSession


# =============================================================================
# Local helpers / config
# =============================================================================

def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class _RunRuntime:
    """
    Mutable runtime for a single active ASR run.

    Kept private to this adapter; the reducer must not depend on any of this.
    """
    run_id: int
    started_ts_ms: int
    recent_audio: list[NDArray[np.float32]]

    last_partial_text: str
    stable_streak: int
    last_partial_is_stable: bool

    endpoint_emitted: bool
    speech_seen: bool
    last_decode_ts_ms: int

    partial_task: Optional[asyncio.Task[None]]
    final_task: Optional[asyncio.Task[None]]


class StreamingASRAdapter:
    """
    Deterministic streaming ASR adapter using WhisperEngine.
    """

    _PARTIAL_DECODE_MIN_INTERVAL_MS = 200
    _SILENCE_RMS_THRESHOLD = 0.02
    _RECENT_AUDIO_TAIL_S = 2.0
    _PRE_ROLL_MS = 300 # TODO: should these be in spec?

    def __init__(
        self,
        *,
        emit_event: Callable[[Event], Coroutine[Any, Any, None]],
        model: str = "base",
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
        prefer_faster_whisper: bool = True,
        session: VoiceSession | None = None,
    ) -> None:
        self._emit_async = emit_event  # Store async callback
        self._engine = WhisperEngine(
            model=model,
            device=device,
            compute_type=compute_type,
            language=language,
            prefer_faster_whisper=prefer_faster_whisper,
        )
        self._active: Optional[_RunRuntime] = None
        self._lock = asyncio.Lock()
        self._session = session

    def _emit(self, event: Event) -> None:
        """
        Emit an event to runtime.

        This is a synchronous convenience wrapper that schedules
        async emission. Adapters call this instead of awaiting.
        """
        asyncio.create_task(self._emit_async(event))

    async def start_stream(self, run_id: int) -> None:
        """
        Begin a new ASR streaming run.

        Semantics:
        - Hard-resets any existing ASR run and associated tasks.
        - Clears all buffered audio and partial state.
        - Initializes runtime state for the given run_id.
        - Subsequent audio frames are accepted only if run_id matches.

        Notes:
        - This does NOT emit any ASR events by itself.
        - This does NOT start decoding until audio frames arrive.
        - Exactly one run may be active at a time.
        """
        print("ASR start_stream BEGIN", {
            "t_ns": time.monotonic_ns(),
            "run_id": run_id,
        })
        async with self._lock:
            # Preserve recent audio across StopASR → StartASR boundary
            preserved_audio = (
                list(self._active.recent_audio) if self._active else []
            )
            await self._hard_reset_locked()
            self._engine.reset()
            self._active = _RunRuntime(
                run_id=run_id,
                started_ts_ms=_now_ms(),
                recent_audio=preserved_audio,
                last_partial_text="",
                stable_streak=0,
                last_partial_is_stable=False,
                endpoint_emitted=False,
                speech_seen=False,
                last_decode_ts_ms=0,
                partial_task=None,
                final_task=None,
            )

            # ------------------------------------------------------------------
            # Inject pre-roll (last PRE_ROLL_MS of preserved audio)
            # ------------------------------------------------------------------
            if preserved_audio:
                pre_roll_samples = int(
                    (self._PRE_ROLL_MS / 1000.0) * AUDIO_SAMPLE_RATE_HZ
                )

                total_samples = sum(arr.shape[0] for arr in preserved_audio)
                frames_to_inject: list[NDArray[np.float32]] = []

                if total_samples > pre_roll_samples:
                    samples_needed = pre_roll_samples
                    for frame in reversed(preserved_audio):
                        if samples_needed <= 0:
                            break
                        if frame.shape[0] <= samples_needed:
                            frames_to_inject.insert(0, frame)
                            samples_needed -= frame.shape[0]
                        else:
                            frames_to_inject.insert(0, frame[-samples_needed:])
                            samples_needed = 0
                else:
                    frames_to_inject = preserved_audio

                injected_samples = sum(f.shape[0] for f in frames_to_inject)
                injected_ms = (injected_samples / AUDIO_SAMPLE_RATE_HZ) * 1000

                print(
                    f"[ASR PRE-ROLL] Injecting {injected_ms:.1f}ms "
                    f"({injected_samples} samples) for run_id={run_id}"
                )

                for f32_frame in frames_to_inject:
                    pcm_bytes = float32_to_pcm16le_mono(f32_frame)
                    self._engine.append_pcm16_frame(pcm_bytes)

                # Critical invariant:
                # After start_stream(), recent_audio must contain ONLY post-start audio
                self._active.recent_audio.clear()
                print("ASR start_stream END", {
                    "t_ns": time.monotonic_ns(),
                    "run_id": run_id,
                })

    async def send_audio(self, run_id: int, sequence_num: int, pcm_bytes: bytes) -> None:
        """
        Ingest a single audio frame for the active ASR run.

        Rules:
        - If run_id does not match the active run, the frame is dropped silently.
        - Audio is appended to the internal Whisper buffer and recent-audio tail.
        - May trigger:
            - Partial ASR decode (best-effort, rate-limited)
            - Endpoint detection via silence
            - Final decode + ASR_FINAL_TIMEOUT scheduling

        Notes:
        - sequence_num is accepted for interface symmetry but not used here.
        - This method performs no blocking ASR work directly.
        - All decoding occurs in background tasks.
        """


        async with self._lock:
            r = self._active
            if r is None or r.run_id != run_id:
                return

            if len(pcm_bytes) != AUDIO_BYTES_PER_FRAME_PCM:
                raise ValueError(
                    f"ASR expected {AUDIO_BYTES_PER_FRAME_PCM} PCM bytes, "
                    f"got {len(pcm_bytes)} (header not stripped?)"
                )

            self._engine.append_pcm16_frame(pcm_bytes)

            f32 = pcm16le_to_float32_mono(pcm_bytes)
            r.recent_audio.append(cast(NDArray[np.float32], f32))
            self._trim_recent_audio_tail_locked(r)

            print(
                "[ASR SEND]",
                {
                    "run_id": run_id,
                    "active_run_id": r.run_id if r else None,
                    "endpoint_emitted": r.endpoint_emitted if r else None,
                    "speech_seen": r.speech_seen if r else None,
                }
            )

            if not r.speech_seen:
                if self._rms(f32) >= self._SILENCE_RMS_THRESHOLD:
                    r.speech_seen = True

            if (not r.endpoint_emitted) and r.speech_seen:
                # DIAGNOSTIC
                if len(r.recent_audio) >= 1 and (sequence_num % 50 == 1):
                    tail = r.recent_audio[-1]
                    rms = float(np.sqrt(np.mean(tail * tail))) if tail.size else 0.0
                    mx = float(np.max(np.abs(tail))) if tail.size else 0.0
                    print(
                        f"[ADAPTER] seq={sequence_num} "
                        f"frame_rms={rms:.6f} "
                        f"frame_max={mx:.6f} "
                        f"speech_seen={r.speech_seen} "
                        f"endpoint={r.endpoint_emitted}"
                    )
                # END DIAGNOSTIC
                if self._is_tail_silent_locked(r): # NOTE come back to this
                    r.endpoint_emitted = True
                    print("ASR END_OF_SPEECH", {
                        "t_ns": time.monotonic_ns(),
                        "run_id": run_id,
                    })
                    self._emit(
                        EndOfSpeech(
                            event_type=EventType.END_OF_SPEECH,
                            ts_ms=_now_ms(),
                            service=Service.ASR,
                            run_id=run_id,
                        )
                    )

                    r.final_task = asyncio.create_task(self._final_decode_task(run_id))

            if not r.endpoint_emitted:
                now = _now_ms()
                if (now - r.last_decode_ts_ms) >= self._PARTIAL_DECODE_MIN_INTERVAL_MS:
                    r.last_decode_ts_ms = now
                    if r.partial_task is None or r.partial_task.done():
                        r.partial_task = asyncio.create_task(
                            self._partial_decode_task(run_id)
                        )

    async def cancel(self, run_id: int) -> None:
        """
        Graceful cancellation with coordination.

        Waits for lock, checks run_id, emits CANCEL_ACK.
        This is the normal path during orchestrator-driven cancellation.
        """
        print("cancelling ASR")
        async with self._lock:
            r = self._active
            if r is None or r.run_id != run_id:
                return
            await self._hard_reset_locked()

            # Emit acknowledgment (normal protocol path)
            self._emit(CancelAck(
                event_type=EventType.CANCEL_ACK,
                service=Service.ASR,
                run_id=run_id,
                ts_ms=_now_ms(),
            ))

    def force_reset(self) -> None:
        """
        Emergency hard reset without coordination.

        Called when cancel ACK timeout expires (adapter is stuck/unresponsive).
        This method must:
        - Not await
        - Not acquire locks
        - Not emit events
        - Leave adapter in a clean idle state
        """

        r = self._active
        self._active = None  # Invalidate immediately

        if r is not None:
            # Best-effort cancellation of scheduled tasks
            for task in (
                r.partial_task,
                r.final_task,
            ):
                if task is not None and not task.done():
                    task.cancel()

       # Reset engine state (if applicable)
        if hasattr(self._engine, 'reset'):
            try:
                self._engine.reset()
            except Exception: # pylint: disable=broad-exception-caught
                pass


    async def _partial_decode_task(self, run_id: int) -> None:
        print("ASR PARTIAL DECODE START", {
            "t_ns": time.monotonic_ns(),
            "run_id": run_id,
        })
        try:
            result_text = await self._transcribe_text_async(
                window_s=None
            ) #TODO: should window_s be _RECENT_AUDIO_TAIL_S or None?
        except WhisperBackendError as e:
            self._emit(
                ASRError(
                    event_type=EventType.ASR_ERROR,
                    ts_ms=_now_ms(),
                    service=Service.ASR,
                    run_id=run_id,
                    reason=str(e),
                )
            )
            return
        except Exception as e: # pylint: disable=broad-exception-caught
            self._emit(
                ASRError(
                    event_type=EventType.ASR_ERROR,
                    ts_ms=_now_ms(),
                    service=Service.ASR,
                    run_id=run_id,
                    reason=f"asr_partial_exception: {e!r}",
                )
            )
            return

        print("ASR PARTIAL DECODE END", {
            "t_ns": time.monotonic_ns(),
            "run_id": run_id,
            "text_len": len(result_text or ""),
        })

        async with self._lock:
            r = self._active
            if r is None or r.run_id != run_id or r.endpoint_emitted:
                return

            text = (result_text or "").strip()
            is_stable = self._update_stability_locked(r, text)

            print("ASR_PARTIAL EMIT", {
                "t_ns": time.monotonic_ns(),
                "run_id": run_id,
            })

            self._emit(
                ASRPartial(
                    event_type=EventType.ASR_PARTIAL,
                    ts_ms=_now_ms(),
                    service=Service.ASR,
                    run_id=run_id,
                    text=text,
                    ts_range=None,
                    is_stable=is_stable,
                )
            )

    async def _final_decode_task(self, run_id: int) -> None:
        print("ASR FINAL DECODE START", {
            "t_ns": time.monotonic_ns(),
            "run_id": run_id,
        })
        try:
            result_text = await self._transcribe_text_async(
                window_s=None
            )
        except WhisperBackendError as e:
            self._emit(
                ASRError(
                    event_type=EventType.ASR_ERROR,
                    ts_ms=_now_ms(),
                    service=Service.ASR,
                    run_id=run_id,
                    reason=str(e),
                )
            )
            return
        except Exception as e: # pylint: disable=broad-exception-caught
            self._emit(
                ASRError(
                    event_type=EventType.ASR_ERROR,
                    ts_ms=_now_ms(),
                    service=Service.ASR,
                    run_id=run_id,
                    reason=f"asr_final_exception: {e!r}",
                )
            )
            return

        async with self._lock:
            r = self._active
            if r is None or r.run_id != run_id:
                return

            text = (result_text or "").strip()

            # Drop empty finals
            if not text:
                print(
                    "[ASR] dropping empty ASRFinal",
                    {"run_id": run_id}
                )
                return

            print("ASR_FINAL EMIT", {
                "t_ns": time.monotonic_ns(),
                "run_id": run_id,
                "text_len": len(text),
            })
            self._emit(
                ASRFinal(
                    event_type=EventType.ASR_FINAL,
                    ts_ms=_now_ms(),
                    service=Service.ASR,
                    run_id=run_id,
                    text=text,
                    ts_range=None,
                )
            )

    async def _transcribe_text_async(self, *, window_s: float | None) -> str:
        loop = asyncio.get_running_loop()

        def _call() -> str:
            res = self._engine.transcribe(
                window_s=window_s,
                sample_rate_hz=AUDIO_SAMPLE_RATE_HZ,
                temperature=0.0,
            )
            return res.text

        return await loop.run_in_executor(None, _call)

    def _trim_recent_audio_tail_locked(self, r: _RunRuntime) -> None:
        max_samples = int(self._RECENT_AUDIO_TAIL_S * AUDIO_SAMPLE_RATE_HZ)
        total = sum(int(x.shape[0]) for x in r.recent_audio)
        if total <= max_samples:
            return

        while r.recent_audio and total > max_samples:
            head = r.recent_audio[0]
            if total - head.shape[0] >= max_samples:
                total -= head.shape[0]
                r.recent_audio.pop(0)
            else:
                keep = max_samples - (total - head.shape[0])
                r.recent_audio[0] = head[-keep:]
                break

    def _is_tail_silent_locked(self, r: _RunRuntime) -> bool:
        window_samples = int((SILENCE_DETECTION_MS / 1000.0) * AUDIO_SAMPLE_RATE_HZ)

        if not r.recent_audio:
            return True

        chunks: list[NDArray[np.float32]] = []
        remaining = window_samples
        for arr in reversed(r.recent_audio):
            if remaining <= 0:
                break
            if arr.shape[0] <= remaining:
                chunks.append(arr)
                remaining -= arr.shape[0]
            else:
                chunks.append(arr[-remaining:])
                remaining = 0
                break

        if remaining > 0:
            return False

        tail = np.concatenate(list(reversed(chunks)), axis=0)
        return self._rms(tail) < self._SILENCE_RMS_THRESHOLD

    def _rms(self, audio_f32: np.ndarray) -> float:
        if audio_f32.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_f32 * audio_f32)))

    def _update_stability_locked(self, r: _RunRuntime, new_text: str) -> bool:
        if new_text and new_text == r.last_partial_text:
            r.stable_streak += 1
        else:
            r.stable_streak = 0
            r.last_partial_text = new_text

        r.last_partial_is_stable = bool(new_text) and (r.stable_streak >= 2)
        return r.last_partial_is_stable

    async def _hard_reset_locked(self) -> None:
        r = self._active
        if r is not None:
            for t in (r.partial_task, r.final_task):
                if t is not None and not t.done():
                    t.cancel()
            r.recent_audio.clear()
        if self._active:
            print(f"ASR: clearing _.active {self._active.run_id}")

        self._engine.reset()
        self._active = None
