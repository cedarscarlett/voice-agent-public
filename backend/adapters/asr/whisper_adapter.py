# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
"""
Whisper ASR engine wrapper (Phase 6).

This module is deliberately "dumb":
- Accepts raw PCM16 (16kHz, mono) audio frames
- Converts to Whisper input format
- Runs transcription
- Returns text (+ optional segment timestamps)

Must NOT:
- Know about run IDs
- Perform endpointing / silence detection
- Emit orchestrator events
- Manage timers
- Make orchestration decisions

It is intentionally usable by the streaming ASR adapter, which *does* own:
buffering, endpointing, run-id gating, cancellation, and event emission.

Implementation notes:
- Whisper is not truly streaming; "partial" here is a best-effort re-decode of the
  current buffer (optionally windowed).
- Supports either `faster-whisper` (preferred) or OpenAI's `whisper` package.

Determinism note:
- Whisper is not bitwise-deterministic across executions.
- Phase 6 guarantees *behavioral determinism* (ordering, gating, cancellation),
  not identical transcripts across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# =============================================================================
# Public result types
# =============================================================================

@dataclass(frozen=True)
class WhisperSegment:
    """
    One timestamped segment of recognized speech.

    Times are in milliseconds relative to the start of the provided audio buffer.
    """
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class WhisperResult:
    """
    Result of a transcription pass.

    text:
        Full recognized text for the provided audio buffer/window.

    segments:
        Optional timestamped segments (may be empty if backend doesn't provide them).

    ts_range_ms:
        Convenience range for the *covered window* (start_ms, end_ms) if windowed.
        For full-buffer decode, this is typically (0, audio_len_ms).
    """
    text: str
    segments: tuple[WhisperSegment, ...] = ()
    ts_range_ms: tuple[int, int] | None = None


# =============================================================================
# PCM helpers
# =============================================================================

def pcm16le_to_float32_mono(pcm_bytes: bytes) -> np.ndarray:
    """
    Convert PCM16 little-endian mono bytes to float32 in [-1.0, 1.0).

    No resampling. No channel mixing. No validation beyond dtype conversion.
    """
    if len(pcm_bytes) % 2 != 0:
        # Truncated sample; caller should treat as malformed frame upstream.
        pcm_bytes = pcm_bytes[: len(pcm_bytes) - 1]

    audio_i16 = np.frombuffer(pcm_bytes, dtype="<i2")  # little-endian int16
    audio_f32 = audio_i16.astype(np.float32) / 32768.0
    return audio_f32

def float32_to_pcm16le_mono(audio_f32: np.ndarray) -> bytes:
    """
    Inverse of pcm16le_to_float32_mono.
    Does not contain a np.int16(32768) == -32768 bug.
    """
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = np.round(audio_f32 * 32767.0).astype(np.int16)
    return audio_i16.tobytes()


# =============================================================================
# Backend selection
# =============================================================================

class WhisperBackendError(RuntimeError):
    """Raised when no Whisper backend is available or a backend call fails."""


class WhisperEngine:
    """
    Minimal Whisper inference wrapper (mechanism only).

    This class is intentionally "dumb":
    - Accepts PCM16 frames, returns text
    - No endpointing, silence detection, or orchestration
    - No run-ID awareness, event emission, or cancellation logic
    - Synchronous by design (caller must handle async via executor)

    Assumptions:
    - Input is PCM16, 16kHz, mono (caller validates)
    - Frame corruption is an upstream bug, not handled here
    - Whisper inference is NOT bitwise-deterministic (temp=0 helps, but not guaranteed)

    Responsibilities pushed to streaming adapter:
    - Buffering policy (when to trim)
    - Silence detection / endpointing
    - Partial stability tracking
    - Run-ID gating / cancellation
    - Event emission to orchestrator
    """

    def __init__(
        self,
        *,
        model: str = "base",
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
        prefer_faster_whisper: bool = True,
    ) -> None:
        self._language = language
        self._buffer: list[np.ndarray] = []

        self._backend_kind: str
        self._backend: Any

        if prefer_faster_whisper:
            try:
                from faster_whisper import WhisperModel  # type: ignore pylint: disable=import-outside-toplevel

                kwargs: dict[str, Any] = {}
                if device is not None:
                    kwargs["device"] = device
                if compute_type is not None:
                    kwargs["compute_type"] = compute_type

                self._backend_kind = "faster_whisper"
                self._backend = WhisperModel(model, **kwargs)
                return
            except Exception: # pylint: disable=broad-exception-caught
                pass

        try:
            import whisper  # type: ignore pylint: disable=import-outside-toplevel

            self._backend_kind = "openai_whisper"
            self._backend = whisper.load_model(model)
        except Exception as e:
            raise WhisperBackendError(
                "No Whisper backend available. Install one of:\n"
                "  - faster-whisper (preferred)\n"
                "  - openai-whisper\n"
                f"Original import error: {e!r}"
            ) from e

    # -------------------------------------------------------------------------
    # Buffer lifecycle
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Drop all buffered audio."""
        self._buffer.clear()

    def total_buffered_audio_seconds(self, *, sample_rate_hz: int = 16000) -> float:
        """
        Total buffered audio duration.

        Note:
            This reflects *all* audio currently stored, regardless of any
            windowing applied during transcription.
        """
        if not self._buffer:
            return 0.0
        n = int(sum(chunk.shape[0] for chunk in self._buffer))
        return float(n) / float(sample_rate_hz)

    def trim_before_seconds(self, seconds: float, *, sample_rate_hz: int = 16000) -> None:
        """
        Trim the internal buffer to keep only the last `seconds` of audio.

        This provides a *mechanism*, not a policy. The streaming adapter decides
        when (and whether) to call this.
        """
        if seconds <= 0:
            self.reset()
            return

        audio = self._concat_audio()
        max_samples = int(seconds * sample_rate_hz)
        if audio.shape[0] > max_samples:
            self._buffer = [audio[-max_samples:]]

    def append_pcm16_frame(self, pcm_bytes: bytes) -> None:
        """
        Append one PCM16 frame to the internal buffer.

        Assumptions (not validated):
        - pcm_bytes is exactly 640 bytes (20ms @ 16kHz)
        - Little-endian signed 16-bit samples
        - Mono channel

        Malformed input will corrupt the buffer silently.
        Caller (streaming adapter) must validate upstream.
        """
        self._buffer.append(pcm16le_to_float32_mono(pcm_bytes))

    def _concat_audio(self) -> np.ndarray:
        if not self._buffer:
            return np.zeros((0,), dtype=np.float32)
        if len(self._buffer) == 1:
            return self._buffer[0]
        return np.concatenate(self._buffer, axis=0)

    # -------------------------------------------------------------------------
    # Transcription
    # -------------------------------------------------------------------------

    def transcribe(
        self,
        *,
        window_s: float | None = None,
        sample_rate_hz: int = 16000,
        prompt: str | None = None,
        temperature: float = 0.0,
    ) -> WhisperResult:
        """
        Transcribe the current buffer (or a trailing window of it).

        Notes:
        - Empty buffer returns empty text (caller should check before calling)
        - Blocking call (100-500ms typical for base model)
        - Non-deterministic across runs even at temperature=0
        - No VAD/endpointing applied (vad_filter=False is intentional)
        """
        audio = self._concat_audio()

        if audio.size == 0:
            return WhisperResult(text="", segments=(), ts_range_ms=(0, 0))

        ts_range_ms: tuple[int, int] | None
        if window_s is not None and window_s > 0:
            max_samples = int(window_s * sample_rate_hz)
            if audio.shape[0] > max_samples:
                audio = audio[-max_samples:]
                ts_range_ms = (0, int(window_s * 1000))
            else:
                ts_range_ms = (0, int((audio.shape[0] / sample_rate_hz) * 1000))
        else:
            ts_range_ms = (0, int((audio.shape[0] / sample_rate_hz) * 1000))

        # DIAGNOSTIC
        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        mx = float(np.max(np.abs(audio))) if audio.size else 0.0
        print(f"[ENGINE] decode_samples={audio.shape[0]} window_s={window_s} rms={rms:.6f} max={mx:.6f}", flush=True)
        # END DIAGNOSTIC


        try:
            if self._backend_kind == "faster_whisper":
                return self._transcribe_faster_whisper(
                    audio=audio,
                    language=self._language,
                    prompt=prompt,
                    temperature=temperature,
                    ts_range_ms=ts_range_ms,
                )
            if self._backend_kind == "openai_whisper":
                return self._transcribe_openai_whisper(
                    audio=audio,
                    language=self._language,
                    prompt=prompt,
                    temperature=temperature,
                    ts_range_ms=ts_range_ms,
                )
            raise WhisperBackendError(f"Unknown backend kind: {self._backend_kind}")
        except Exception as e:
            raise WhisperBackendError(f"Whisper transcription failed: {e!r}") from e

    # -------------------------------------------------------------------------
    # Backend-specific implementations
    # -------------------------------------------------------------------------

    def _transcribe_faster_whisper(
        self,
        *,
        audio: np.ndarray,
        language: str | None,
        prompt: str | None,
        temperature: float,
        ts_range_ms: tuple[int, int] | None,
    ) -> WhisperResult:
        kwargs: dict[str, Any] = {
            "language": language,
            "beam_size": 1,
            "temperature": temperature,
            "vad_filter": False,  # endpointing is NOT this module's job
        }
        if prompt:
            kwargs["initial_prompt"] = prompt

        segments_iter, _info = self._backend.transcribe(audio, **kwargs)

        segments: list[WhisperSegment] = []
        text_parts: list[str] = []

        for seg in segments_iter:
            start_ms = int(seg.start * 1000)
            end_ms = int(seg.end * 1000)
            raw = getattr(seg, "text", "")
            seg_text = str(raw).strip()
            if seg_text:
                text_parts.append(seg_text)
            segments.append(WhisperSegment(start_ms=start_ms, end_ms=end_ms, text=seg_text))

        full_text = " ".join(text_parts).strip()
        return WhisperResult(text=full_text, segments=tuple(segments), ts_range_ms=ts_range_ms)

    def _transcribe_openai_whisper(
        self,
        *,
        audio: np.ndarray,
        language: str | None,
        prompt: str | None,
        temperature: float,
        ts_range_ms: tuple[int, int] | None,
    ) -> WhisperResult:
        options: dict[str, Any] = {
            "language": language,
            "task": "transcribe",
            "temperature": temperature,
            "fp16": False,
        }
        if prompt:
            options["initial_prompt"] = prompt

        result = self._backend.transcribe(audio, **options)
        full_text = (result.get("text") or "").strip()

        segments: list[WhisperSegment] = []
        for seg in result.get("segments") or []:
            start_ms = int(float(seg.get("start", 0.0)) * 1000)
            end_ms = int(float(seg.get("end", 0.0)) * 1000)
            seg_text = (seg.get("text") or "").strip()
            segments.append(WhisperSegment(start_ms=start_ms, end_ms=end_ms, text=seg_text))

        return WhisperResult(text=full_text, segments=tuple(segments), ts_range_ms=ts_range_ms)
