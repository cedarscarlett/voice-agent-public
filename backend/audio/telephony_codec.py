import audioop
import numpy as np
from scipy import signal


class TelephonyCodec:
    """Handles bidirectional telephony audio conversion."""

    def __init__(self):
        # No state needed - scipy resampling is stateless
        pass

    def decode(self, mulaw_bytes: bytes) -> bytes:
        """Convert μ-law@8kHz to PCM16@16kHz."""
        if not mulaw_bytes:
            return b""

        # μ-law → PCM 8kHz
        pcm8k = audioop.ulaw2lin(mulaw_bytes, 2)

        # Convert to numpy
        samples_8k = np.frombuffer(pcm8k, dtype=np.int16)

        # Upsample 8kHz → 16kHz (2x)
        samples_16k = signal.resample_poly(samples_8k, 2, 1)

        # Clip and convert back to int16
        samples_16k = np.clip(samples_16k, -32768, 32767).astype(np.int16)

        return samples_16k.tobytes()

    def encode(self, pcm16k_bytes: bytes) -> bytes:
        """Convert PCM16@16kHz to μ-law@8kHz."""
        if not pcm16k_bytes:
            return b""

        # Convert to numpy
        samples_16k = np.frombuffer(pcm16k_bytes, dtype=np.int16)

        # Downsample 16kHz → 8kHz (1/2)
        samples_8k = signal.resample_poly(samples_16k, 1, 2)

        # Clip and convert back to int16
        samples_8k = np.clip(samples_8k, -32768, 32767).astype(np.int16)
        pcm8k = samples_8k.tobytes()

        # PCM 8kHz → μ-law
        mulaw = audioop.lin2ulaw(pcm8k, 2)

        return mulaw
