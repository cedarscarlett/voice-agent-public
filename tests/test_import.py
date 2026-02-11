"""
ElevenLabs TTS adapter (Phase 8).

Implements a chunked, non-streaming Text-to-Speech adapter using the
ElevenLabs streaming TTS API.

Role in the system:
- Receives *pre-chunked* text segments from the orchestrator.
- Performs one TTS synthesis call per chunk.
- Converts provider output to PCM16 16kHz mono.
- Emits exactly one terminal TTS event per chunk:
    - TTSChunkComplete(run_id, chunk_index, pcm_bytes), or
    - TTSError(run_id, reason).

Architectural constraints:
- Chunking policy is orchestrator-owned and deterministic.
- Audio framing (20ms frames) is NOT handled upstream.
- No retries, timers, or backpressure logic live here.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

from elevenlabs.client import AsyncElevenLabs

from orchestrator.enums.service import Service
from orchestrator.events import (
    Event,
    EventType,
    TTSChunkComplete,
    TTSError,
    TTSAudioFrame,
)

from spec import AUDIO_BYTES_PER_FRAME_PCM


class ElevenLabsTTSAdapter:
    """
    ElevenLabs chunked TTS adapter.

    Design:
    - One asyncio task per (run_id, chunk_index)
    - Fire-and-forget by design
    - Event-driven output identical to Speechmatics adapter
    """

    def __init__(
        self,
        *,
        emit_event: Callable[[Event], Awaitable[None]],
        api_key: str,
        session_id: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # default ElevenLabs voice
        model_id: str = "eleven_turbo_v2",
    ) -> None:
        self._emit_event = emit_event
        self._api_key = api_key
        self._session_id = session_id
        self._voice_id = voice_id
        self._model_id = model_id

        self._client = AsyncElevenLabs(api_key=api_key)

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
        voice: str = "default",
    ) -> None:
        key = (run_id, chunk_index)

        if key in self._tasks:
            return

        task = asyncio.create_task(
            self._run_synthesis(
                run_id=run_id,
                chunk_index=chunk_index,
                text=text,
            )
        )

        self._tasks[key] = task

        def _cleanup(_: asyncio.Task[None]) -> None:
            self._tasks.pop(key, None)

        task.add_done_callback(_cleanup)

    async def cancel(self, run_id: int) -> None:
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
    ) -> None:
        try:
            t0 = time.monotonic_ns()

            audio_stream = await self._client.text_to_speech.stream(
                voice_id=self._voice_id,
                model_id=self._model_id,
                text=text,
                output_format="pcm_16000",
            )

            frame_buffer = b""
            sequence = 0

            async for chunk in audio_stream:
                if not chunk:
                    continue

                frame_buffer += chunk

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

        except asyncio.CancelledError:
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

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)
