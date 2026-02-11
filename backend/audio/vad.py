"""
A minimal, energy-based Voice Activity Detection (VAD) module.

Provides a simple RMS-energy threshold VAD intended for
real-time or streaming audio pipelines. It operates on short, fixed-size
audio frames (float32 samples) and reports voice activity only after a
configurable number of consecutive frames exceed a given energy threshold.
"""
import numpy as np

class EnergyVAD:
    """
    Simple energy-based Voice Activity Detector (VAD).

    This VAD operates on short frames of audio represented as float32 samples.
    For each observed frame, it computes the RMS (root-mean-square) energy and
    compares it against a fixed threshold. Voice activity is considered present
    only after a configurable number of *consecutive* frames exceed the threshold.

    This design provides basic temporal smoothing and avoids triggering on
    single-frame noise spikes.
    """
    def __init__(self, threshold: float, frames_required: int):
        self._threshold = threshold
        self._frames_required = frames_required
        self._count = 0

    def observe(self, f32: np.ndarray) -> bool:
        """
        Observe a single audio frame and update VAD state.

        The frame's RMS energy is computed and compared to the configured
        threshold. If the frame is above threshold, an internal counter of
        consecutive active frames is incremented; otherwise, the counter is
        reset to zero.

        Args:
            f32:
                A 1D NumPy array of float32 audio samples representing one
                analysis frame.

        Returns:
            True if voice activity has been detected, meaning that at least
            `frames_required` consecutive frames (including this one) have
            exceeded the energy threshold. False otherwise.
        """
        print("VAD observing")
        rms = float(np.sqrt(np.mean(np.square(f32))))
        if rms >= self._threshold:
            self._count += 1
        else:
            self._count = 0
        if self._count >= self._frames_required:
            print("VAD returning True")
        return self._count >= self._frames_required

    def reset(self) -> None:
        """
        Reset the internal VAD state.

        This clears the count of consecutive above-threshold frames, causing
        subsequent detection to require a fresh run of `frames_required`
        qualifying frames.
        """
        self._count = 0
