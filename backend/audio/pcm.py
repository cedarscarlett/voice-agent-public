"""PCM conversion utilities."""
import numpy as np

def pcm16le_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """
    Convert PCM16 little-endian mono bytes to float32 in [-1.0, 1.0).

    Runtime-safe, adapter-agnostic utility.
    No resampling. No channel mixing.
    """
    if len(pcm_bytes) % 2 != 0:
        # Truncated sample; caller should treat as malformed frame upstream.
        pcm_bytes = pcm_bytes[: len(pcm_bytes) - 1]

    audio_i16 = np.frombuffer(pcm_bytes, dtype="<i2")  # little-endian int16
    audio_f32 = audio_i16.astype(np.float32) / 32768.0
    return audio_f32
