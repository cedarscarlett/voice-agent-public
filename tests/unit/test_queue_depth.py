# pylint: disable=missing-module-docstring,missing-function-docstring

import time

from audio.frames import AudioFrame
from audio.queues import AudioFrameQueue
from spec import AUDIO_FRAME_DURATION_S


def make_frame(seq: int) -> AudioFrame:
    return AudioFrame(
        sequence_num=seq,
        pcm_bytes=b"\x00\x00" * 320,
        ts_ms=int(time.time() * 1000),
    )


# ---------------------------------------------------------------------
# depth_seconds math
# ---------------------------------------------------------------------

def test_depth_seconds_exact():
    q = AudioFrameQueue(max_depth_s=1.0)

    q.enqueue(make_frame(1))
    q.enqueue(make_frame(2))
    q.enqueue(make_frame(3))

    expected = 3 * AUDIO_FRAME_DURATION_S
    assert q.depth_seconds() == expected


# ---------------------------------------------------------------------
# Overflow behavior
# ---------------------------------------------------------------------

def test_overflow_drops_newest():
    max_depth = 2 * AUDIO_FRAME_DURATION_S
    q = AudioFrameQueue(max_depth_s=max_depth)

    assert q.enqueue(make_frame(1)) is True
    assert q.enqueue(make_frame(2)) is True

    # Would exceed max_depth_s
    assert q.enqueue(make_frame(3)) is False

    assert q.drops.overflow == 1
    assert q.total_drops() == 1
    assert q.depth_seconds() == max_depth


# ---------------------------------------------------------------------
# DEGRADED drop behavior
# ---------------------------------------------------------------------

def test_degraded_drops_oldest():
    q = AudioFrameQueue(max_depth_s=10.0)

    q.enqueue(make_frame(1))
    q.enqueue(make_frame(2))
    q.enqueue(make_frame(3))

    # DEGRADED enqueue should drop OLDEST (seq=1)
    assert q.enqueue(make_frame(4), degraded=True) is True

    assert q.drops.degraded == 1
    assert q.total_drops() == 1

    # Oldest frame should now be seq=2
    head = q.peek()
    assert head is not None
    assert head.sequence_num == 2


def test_degraded_and_overflow_accounted_separately():
    max_depth = 2 * AUDIO_FRAME_DURATION_S
    q = AudioFrameQueue(max_depth_s=max_depth)

    q.enqueue(make_frame(1))
    q.enqueue(make_frame(2))

    # Overflow
    q.enqueue(make_frame(3))
    # Degraded drop
    q.enqueue(make_frame(4), degraded=True)

    assert q.drops.overflow == 1
    assert q.drops.degraded == 1
    assert q.total_drops() == 2
