import audioop
import wave
from pathlib import Path


INPUT_PCM = "tts.pcm"       # PCM16 mono @ 16kHz
OUTPUT_WAV = "roundtrip.wav"


def pcm16k_to_mulaw8k(pcm16k_bytes: bytes) -> bytes:
    # Normalize to telephony level
    peak16 = audioop.max(pcm16k_bytes, 2)
    if peak16 > 0:
        target_peak = 6000
        gain = min(1.0, target_peak / peak16)
        pcm16k_bytes = audioop.mul(pcm16k_bytes, 2, gain)

    pcm8k, _ = audioop.ratecv(
        pcm16k_bytes,
        2,
        1,
        16000,
        8000,
        None,
    )

    mulaw = audioop.lin2ulaw(pcm8k, 2)
     # Measure quantization damage
    import struct
    pcm8k_decoded = audioop.ulaw2lin(mulaw, 2)

    original = struct.unpack(f'<{len(pcm8k)//2}h', pcm8k)
    decoded = struct.unpack(f'<{len(pcm8k_decoded)//2}h', pcm8k_decoded)

    errors = [abs(o - d) for o, d in zip(original, decoded)]
    max_error = max(errors)
    avg_error = sum(errors) / len(errors)

    print(f"μ-law quantization error: max={max_error}, avg={avg_error:.1f}")
    return mulaw


def mulaw8k_to_pcm16k(mulaw_bytes: bytes) -> bytes:
    pcm8k = audioop.ulaw2lin(mulaw_bytes, 2)

    pcm16k, _ = audioop.ratecv(
        pcm8k,
        2,
        1,
        8000,
        16000,
        None,
    )

    return pcm16k


def main():
    pcm_path = Path(INPUT_PCM)
    pcm16k = pcm_path.read_bytes()

    mulaw = pcm16k_to_mulaw8k(pcm16k)
    pcm16k_roundtrip = mulaw8k_to_pcm16k(mulaw)

    with wave.open(OUTPUT_WAV, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm16k_roundtrip)

    print("Done.")
    print(f"Input PCM bytes: {len(pcm16k)}")
    print(f"μ-law bytes: {len(mulaw)}")
    print(f"Output WAV: {OUTPUT_WAV}")


if __name__ == "__main__":
    main()
