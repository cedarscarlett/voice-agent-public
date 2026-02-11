# backend/audio/queues.py
"""
Bounded audio frame queues with canonical depth measurement.

Spec §10 requirements:
- Depth measured in seconds (not frame count)
- Explicit drop behavior
- Drop reasons distinguishable (overflow vs DEGRADED)
- Drop OLDEST frames when DEGRADED to prevent latency buildup
- Deterministic, synchronous behavior
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

from audio.frames import AudioFrame
from spec import AUDIO_FRAME_DURATION_S


class DropReason(str, Enum):
    """
    Reason an audio frame was dropped.
    """
    OVERFLOW = "overflow"
    DEGRADED = "degraded"


@dataclass
class DropCounters:
    """
    Drop counters for observability.
    """
    overflow: int = 0
    degraded: int = 0


class AudioFrameQueue:
    """
    Bounded FIFO queue for AudioFrame objects.

    Drop rules (spec §10):
    - degraded=True: drop OLDEST frame to make room, then enqueue new
    - else: drop NEW frame if enqueue would exceed max_depth_s
    """

    def __init__(self, *, max_depth_s: float) -> None:
        if max_depth_s <= 0:
            raise ValueError("max_depth_s must be > 0")

        self._max_depth_s: float = max_depth_s
        self._frames: Deque[AudioFrame] = deque()
        self.drops: DropCounters = DropCounters()

    # -------------------------
    # Core queue operations
    # -------------------------

    def enqueue(self, frame: AudioFrame, *, degraded: bool = False) -> bool:
        """
        Enqueue an AudioFrame.

        Returns:
            True if enqueued
            False if dropped
        """
        # DEGRADED mode: drop OLDEST to keep audio fresh
        if degraded and self._frames:
            self._frames.popleft()
            self.drops.degraded += 1
            # fall through and attempt enqueue

        # Normal overflow protection
        if self.depth_seconds() + AUDIO_FRAME_DURATION_S > self._max_depth_s:
            self.drops.overflow += 1
            return False

        self._frames.append(frame)
        return True

    def dequeue(self) -> Optional[AudioFrame]:
        """
        Dequeue the oldest AudioFrame.

        Returns None if queue is empty.
        """
        if not self._frames:
            return None
        return self._frames.popleft()

    def peek(self) -> Optional[AudioFrame]:
        """
        View the oldest frame without removing it.

        Used for sequence gap detection and observability.
        """
        return self._frames[0] if self._frames else None

    def clear(self) -> None:
        """
        Drop all queued frames without counting them as drops.

        Used during hard resets / cancellations.
        """
        self._frames.clear()

    # -------------------------
    # Introspection helpers
    # -------------------------

    def __len__(self) -> int:
        return len(self._frames)

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return not self._frames

    def depth_seconds(self) -> float:
        """
        Canonical queue depth in seconds.

        depth_s = num_frames × AUDIO_FRAME_DURATION_S
        """
        return len(self._frames) * AUDIO_FRAME_DURATION_S

    def total_drops(self) -> int:
        """
        Total frames dropped for any reason.
        """
        return self.drops.overflow + self.drops.degraded

    def snapshot(self) -> dict[str, float | int]:
        """
        Lightweight snapshot for logging / metrics.
        """
        return {
            "frames": len(self._frames),
            "depth_s": self.depth_seconds(),
            "dropped_overflow": self.drops.overflow,
            "dropped_degraded": self.drops.degraded,
            "dropped_total": self.total_drops(),
        }

    def peek_latest(self) -> Optional[AudioFrame]:
        """
        View the most recently enqueued frame without removing it.

        Used for observation-only consumers (e.g. VAD).
        """
        return self._frames[-1] if self._frames else None
