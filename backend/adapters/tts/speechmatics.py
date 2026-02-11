"""
Speechmatics TTS adapter (Phase 8).

Implements a chunked, non-streaming Text-to-Speech adapter using the
Speechmatics Async TTS API.

Role in the system:
- Receives *pre-chunked* text segments from the orchestrator.
- Performs one TTS synthesis call per chunk.
- Converts provider output to PCM16 16kHz mono.
- Emits exactly one terminal TTS event per chunk:
    - TTSChunkComplete(run_id, chunk_index, pcm_bytes), or
    - TTSError(run_id, reason).

Architectural constraints:
- Chunking policy is orchestrator-owned and deterministic.
- Audio framing (20ms frames) is NOT handled here.
- No retries, timers, or backpressure logic live in this adapter.
- No state machine transitions or orchestration decisions.
- No direct interaction with WebSocket or UI layers.

Concurrency & cancellation:
- One asyncio task is created per (run_id, chunk_index).
- Cancellation is best-effort and idempotent.
- On cancellation, the adapter may terminate silently without emitting
  a terminal event (allowed by spec).

Event model:
- All outputs are emitted asynchronously via the provided emit_event callback.
- This adapter never returns audio data directly.

This module intentionally contains provider-specific logic only.
All control flow, policy, and lifecycle management live upstream.
"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

from speechmatics.tts import AsyncClient, OutputFormat, Voice # pyright: ignore[reportMissingTypeStubs] # pylint: disable=no-name-in-module, import-error

from orchestrator.enums.service import Service
from orchestrator.events import (
    Event,
    EventType,
    TTSChunkComplete,
    TTSError,
    TTSAudioFrame
)

from spec import PROVIDER_CHUNK_SIZE, AUDIO_BYTES_PER_FRAME_PCM


class SpeechmaticsTTSAdapter:
    """
    Speechmatics chunked (non-streaming) TTS adapter.

    Design:
    - One asyncio task per (run_id, chunk_index)
    - Fire-and-forget by design: synthesize_chunk schedules work and returns
    - All output delivered via events
    """

    _VOICE_MAP: dict[str, Voice] = {
        "sarah": Voice.SARAH,
        "theo": Voice.THEO,
        "megan": Voice.MEGAN,
    }

    def __init__(
        self,
        *,
        emit_event: Callable[[Event], Awaitable[None]],
        api_key: str,
        session_id: str,
    ) -> None:
        self._emit_event = emit_event
        self._api_key = api_key
        self._session_id = session_id

        # Active synthesis tasks keyed by (run_id, chunk_index)
        self._tasks: dict[tuple[int, int], asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # Public API (TTSAdapter contract)
    # ------------------------------------------------------------------

    async def synthesize_chunk(
        self,
        *,
        run_id: int,
        chunk_index: int,
        text: str,
        voice: str = "sarah",
    ) -> None:
        """
        Schedule synthesis of a single pre-chunked text segment.

        Fire-and-forget:
        - Returns immediately
        - Emits exactly one terminal event asynchronously
        """

        key = (run_id, chunk_index)
        if key in self._tasks:
            # Idempotent safety: duplicate request → ignore
            return

        resolved_voice = self._resolve_voice(voice)

        task = asyncio.create_task(
            self._run_synthesis(
                run_id=run_id,
                chunk_index=chunk_index,
                text=text,
                voice=resolved_voice,
            )
        )
        self._tasks[key] = task

        def _cleanup(_: asyncio.Task[None]) -> None:
            self._tasks.pop(key, None)

        task.add_done_callback(_cleanup)

    async def cancel(self, run_id: int) -> None:
        """
        Best-effort cancellation of all in-flight chunks for a run_id.
        """
        to_cancel = [
            task
            for (rid, _), task in self._tasks.items()
            if rid == run_id
        ]

        for task in to_cancel:
            task.cancel()

        for task in to_cancel:
            try:
                await task
            except asyncio.CancelledError:
                pass

    def force_reset(self) -> None:
        """
        Hard reset: cancel *all* in-flight tasks and clear internal state.
        Synchronous by design (called from watchdog paths).
        """
        for task in self._tasks.values():
            task.cancel()

        self._tasks.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run_synthesis(
        self,
        *,
        run_id: int,
        chunk_index: int,
        text: str,
        voice: Voice,
    ) -> None:
        """
        Internal synthesis task.

        Emits exactly one terminal event:
        - TTSChunkComplete OR
        - TTSError
        """
        try:
            t0 = time.monotonic_ns()

            async with AsyncClient(api_key=self._api_key) as client:
                async with await client.generate(
                    text=text,
                    voice=voice,
                    output_format=OutputFormat.RAW_PCM_16000,
                ) as response:
                    carry = b""
                    frame_buffer = b""
                    sequence = 0

                    async for chunk in response.content.iter_chunked(PROVIDER_CHUNK_SIZE):
                        data = carry + chunk

                        if len(data) % 2 == 1:
                            carry = data[-1:]
                            data = data[:-1]
                        else:
                            carry = b""

                        frame_buffer += data

                        while len(frame_buffer) >= AUDIO_BYTES_PER_FRAME_PCM:
                            frame = frame_buffer[:AUDIO_BYTES_PER_FRAME_PCM]
                            frame_buffer = frame_buffer[AUDIO_BYTES_PER_FRAME_PCM:]

                            if sequence == 0:
                                print(
                                    "FIRST_FRAME_OF_CHUNK",
                                    {
                                        "run_id": run_id,
                                        "chunk_index": chunk_index,
                                        "t_ns": time.monotonic_ns(),
                                    },
                                )

                            await self._emit_event(
                                TTSAudioFrame(
                                    event_type=EventType.TTS_AUDIO_FRAME,
                                    ts_ms=self._now_ms(),
                                    service=Service.TTS,
                                    run_id=run_id,
                                    sequence_num=sequence,
                                    pcm_bytes=frame,
                                )
                            )
                            sequence += 1



            t1 = time.monotonic_ns()
            print(
                "TTS_SYNTH_METRICS",
                {
                    "run_id": run_id,
                    "chunk_index": chunk_index,
                    "chars": len(text),
                    "synth_ns": t1 - t0,
                },
            )


            print(
                "ADAPTER_COMPLETE",
                {
                    "run_id": run_id,
                    "chunk_index": chunk_index,
                    "t_ns": time.monotonic_ns(),
                },
            )

            await self._emit_event(
                TTSChunkComplete(
                    event_type=EventType.TTS_CHUNK_COMPLETE,
                    ts_ms=self._now_ms(),
                    service=Service.TTS,
                    run_id=run_id,
                    chunk_index=chunk_index,
                    pcm_bytes=b"",
                )
            )

            print(
                "CHUNK_COMPLETE",
                {
                    "run_id": run_id,
                    "chunk_index": chunk_index,
                    "t_ns": time.monotonic_ns(),
                },
            )

        except asyncio.CancelledError:
            # Expected during barge-in or explicit cancellation
            # Spec allows silent termination
            pass

        except Exception as exc:  # pylint: disable=broad-exception-caught
            await self._emit_event(
                TTSError(
                    event_type=EventType.TTS_ERROR,
                    ts_ms=self._now_ms(),
                    service=Service.TTS,
                    run_id=run_id,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_voice(cls, voice: str) -> Voice:
        """
        Convert user-facing voice string to Speechmatics Voice enum.

        Defaults to SARAH if unknown.
        """
        return cls._VOICE_MAP.get(voice.lower(), Voice.SARAH)

    @staticmethod
    def _now_ms() -> int:
        """Wall-clock timestamp in milliseconds (coarse)."""
        return int(time.time() * 1000)
