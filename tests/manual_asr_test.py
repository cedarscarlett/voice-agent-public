# tools/manual_asr_streaming_test.py
from __future__ import annotations
import numpy as np
import argparse
import asyncio
import sys
import wave
from dataclasses import dataclass
from typing import Optional

from orchestrator.events import (
    ASRPartial,
    ASRFinal,
    ASRFinalTimeout,
    EndOfSpeech,
    ASRError,
    Event,
)
from adapters.asr.streaming import StreamingASRAdapter


FRAME_SAMPLES = 320          # 20ms @ 16kHz
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 => 2 bytes
REALTIME_SLEEP_S = 0.020


@dataclass
class Counters:
    partials: int = 0
    finals: int = 0
    timeouts: int = 0
    eos: int = 0
    errors: int = 0


def _print_engine_config(adapter: StreamingASRAdapter) -> None:
    """
    Best-effort introspection to confirm GPU configuration.
    This is intentionally tolerant: if WhisperEngine doesn't expose fields, we don't fail.
    """
    engine = getattr(adapter, "_engine", None)
    if engine is None:
        print("[asr] NOTE: could not introspect engine (no adapter._engine).", file=sys.stderr)
        return

    # Common patterns you might have in WhisperEngine.
    device = getattr(engine, "device", None) or getattr(engine, "_device", None)
    backend = getattr(engine, "backend", None) or getattr(engine, "_backend_name", None)
    compute_type = getattr(engine, "compute_type", None) or getattr(engine, "_compute_type", None)
    model = getattr(engine, "model", None) or getattr(engine, "_model", None)

    parts = []
    if model:
        parts.append(f"model={model}")
    if backend:
        parts.append(f"backend={backend}")
    if device:
        parts.append(f"device={device}")
    if compute_type:
        parts.append(f"compute_type={compute_type}")

    if parts:
        print(f"[asr] engine: " + ", ".join(parts), file=sys.stderr)
    else:
        print("[asr] NOTE: engine config not introspectable; relying on constructor args.", file=sys.stderr)


async def _feed_wav_frames(
    *,
    adapter: StreamingASRAdapter,
    wav_path: str,
    run_id: int,
    realtime: bool,
    start_seq: int = 1,
) -> None:
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        ch = wf.getnchannels()

        if sr != 16000 or sw != 2 or ch != 1:
            raise RuntimeError(
                f"WAV must be 16kHz mono PCM16. Got sr={sr}, sampwidth={sw}, channels={ch}"
            )

        seq = start_seq
        while True:
            pcm = wf.readframes(FRAME_SAMPLES)  # returns bytes length == FRAME_BYTES (unless EOF)
            if not pcm:
                break
            if len(pcm) != FRAME_BYTES:
                # If the WAV ends mid-frame, pad with zeros (silence).
                pcm = pcm + (b"\x00" * (FRAME_BYTES - len(pcm)))

            # DIAGNOSTIC
            audio_i16 = np.frombuffer(pcm, dtype="<i2")
            mx = int(np.max(np.abs(audio_i16))) if audio_i16.size else 0
            if seq % 50 == 1:  # about once per second
                print(f"[FEED] seq={seq} max_i16={mx} bytes={len(pcm)}")
            # END DIAGNOSTIC

            await adapter.send_audio(run_id, seq, pcm)
            seq += 1

            if realtime:
                await asyncio.sleep(REALTIME_SLEEP_S)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Path to 16kHz mono PCM16 WAV")
    ap.add_argument("--model", default="base", help="Whisper model name (e.g. base, small, medium)")
    ap.add_argument("--device", default="cuda", help="Device for WhisperEngine (use cuda for GPU)")
    ap.add_argument(
        "--compute-type",
        default=None,
        help="faster-whisper compute_type (e.g. float16, int8_float16). Leave None to use engine default.",
    )
    ap.add_argument("--language", default=None, help="Language hint (e.g. en).")
    ap.add_argument("--no-faster-whisper", action="store_true", help="Force OpenAI whisper backend if supported.")
    ap.add_argument("--realtime", action="store_true", help="Sleep 20ms per frame (more realistic timing).")
    ap.add_argument("--run-id", type=int, default=1, help="ASR run_id")
    args = ap.parse_args()

    counters = Counters()
    done_evt = asyncio.Event()
    last_partial: Optional[str] = None

    def emit_event(ev: Event) -> None:
        nonlocal last_partial
        # Print a compact event trace
        if isinstance(ev, ASRPartial):
            counters.partials += 1
            last_partial = ev.text
            stable = " stable" if ev.is_stable else ""
            print(f"[{ev.ts_ms}] ASR_PARTIAL{stable}: {ev.text!r}")
        elif isinstance(ev, EndOfSpeech):
            counters.eos += 1
            print(f"[{ev.ts_ms}] END_OF_SPEECH (run_id={ev.run_id})")
        elif isinstance(ev, ASRFinal):
            counters.finals += 1
            print(f"[{ev.ts_ms}] ASR_FINAL: {ev.text!r}")
            done_evt.set()
        elif isinstance(ev, ASRFinalTimeout):
            counters.timeouts += 1
            print(f"[{ev.ts_ms}] ASR_FINAL_TIMEOUT: last_partial={ev.last_partial_text!r} stable={ev.last_partial_stable}")
            done_evt.set()
        elif isinstance(ev, ASRError):
            counters.errors += 1
            print(f"[{ev.ts_ms}] ASR_ERROR: {ev.reason}", file=sys.stderr)
            done_evt.set()
        else:
            print(f"[{getattr(ev, 'ts_ms', '?')}] {type(ev).__name__}: {ev!r}")

    adapter = StreamingASRAdapter(
        emit_event=emit_event,
        model=args.model,
        device=args.device,                 # <-- GPU if cuda
        compute_type=args.compute_type,     # <-- helps guarantee GPU-friendly path in faster-whisper
        language=args.language,
        prefer_faster_whisper=not args.no_faster_whisper,
    )

    _print_engine_config(adapter)

    run_id = args.run_id
    await adapter.start_stream(run_id)

    feeder = asyncio.create_task(
        _feed_wav_frames(
            adapter=adapter,
            wav_path=args.wav,
            run_id=run_id,
            realtime=args.realtime,
        )
    )

    # Give the adapter time to endpoint (silence) and either produce FINAL or FINAL_TIMEOUT.
    # Your adapter emits END_OF_SPEECH then starts final decode + timeout task.
    try:
        await asyncio.wait_for(done_evt.wait(), timeout=15.0)
    except asyncio.TimeoutError:
        print("[asr] TIMEOUT waiting for ASR_FINAL/ASR_FINAL_TIMEOUT.", file=sys.stderr)
    finally:
        await feeder
        await adapter.cancel(run_id)

    print(
        f"[asr] done: partials={counters.partials}, eos={counters.eos}, finals={counters.finals}, "
        f"timeouts={counters.timeouts}, errors={counters.errors}",
        file=sys.stderr,
    )

    # Success if we got a final or timeout without errors.
    if counters.errors > 0:
        return 2
    if counters.finals > 0 or counters.timeouts > 0:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
