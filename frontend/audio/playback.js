/**
 * frontend/audio/playback.js
 *
 * Streaming PCM16 playback (16kHz, mono) with HARD STOP.
 *
 * Phase 5 responsibilities:
 * - Accept inbound 20ms PCM16 frames (640 bytes)
 * - Play immediately with minimal buffering
 * - Provide hard stop() that cancels audio quickly (barge-in)
 * - Provide pause/resume hooks
 * - Emit observability events for latency + sequencing
 *
 * NOT responsible for:
 * - WebSocket receiving/decoding
 * - Barge-in detection
 * - Orchestration decisions
 *
 * Usage:
 *
 *   const playback = new Playback({
 *     onEvent: (evt) => console.log("Playback:", evt),
 *   });
 *
 *   await playback.start();
 *
 *   playback.enqueueFrame({
 *     seqNum,
 *     pcmBytes,
 *     receivedAt: performance.now(),
 *   });
 *
 *   await playback.hardStopAndRestart(); // on barge-in
 */

import {
  AUDIO_SAMPLE_RATE_HZ,
  AUDIO_CHANNELS,
  AUDIO_SAMPLES_PER_FRAME,
  AUDIO_BYTES_PER_FRAME_PCM,
  AUDIO_FRAME_DURATION_S,
  SEQ_NUM_MAX,
  SEQ_NUM_START,
} from "./audio_format.js";

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

function pcm16BytesToFloat32(pcmBytes) {
  const i16 = new Int16Array(
    pcmBytes.buffer,
    pcmBytes.byteOffset,
    pcmBytes.byteLength / 2
  );

  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) {
    f32[i] = i16[i] / 0x8000;
  }
  return f32;
}

// -----------------------------------------------------------------------------
// Playback
// -----------------------------------------------------------------------------

export class Playback {
  constructor({ onEvent } = {}) {
    /**
     * Optional observability hook:
     * onEvent({ type, ts, ...fields })
     */
    this.onEvent = onEvent || null;

    this.audioContext = null;

    this.running = false;
    this.paused = false;

    // FIFO queue of decoded frames
    this._queue = [];

    // AudioContext time cursor (seconds)
    this._scheduledUntil = 0;

    // Expected sequence number (debug only)
    this._nextSeqExpected = null;

    // Hard-stop guard:
    // Incremented on stop() to invalidate any scheduled callbacks
    this._epoch = 0;

    // Stats
    this._framesEnqueued = 0;
    this._framesScheduled = 0;
    this._framesDropped = 0;
  }

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  async start() {
    if (this.running) return;

    this.audioContext = new AudioContext({ sampleRate: AUDIO_SAMPLE_RATE_HZ });

    if (this.audioContext.sampleRate !== AUDIO_SAMPLE_RATE_HZ) {
      console.warn("[playback] unexpected output sample rate", {
        requested: AUDIO_SAMPLE_RATE_HZ,
        actual: this.audioContext.sampleRate,
      });
      // Continue anyway: browser will resample internally.
      // If audio sounds wrong, investigate this first.
    }

    this.running = true;
    this.paused = false;
    this._scheduledUntil = this.audioContext.currentTime;

    this._emit("PLAYBACK_STARTED", {
      sampleRate: this.audioContext.sampleRate,
    });
    console.info("[playback] started", {
      sampleRate: this.audioContext.sampleRate,
    });
  }

  stop() {
    /**
     * HARD STOP:
     * - Invalidate scheduled callbacks via epoch bump
     * - Clear queue
     * - Close AudioContext (forces immediate silence)
     */
    if (!this.running) return;

    this._epoch++;
    this._queue = [];
    this._scheduledUntil = 0;
    this._nextSeqExpected = null;

    this._emit("PLAYBACK_STOPPED", {
      framesEnqueued: this._framesEnqueued,
      framesScheduled: this._framesScheduled,
      framesDropped: this._framesDropped,
    });

    try {
      this.audioContext.close();
    } catch {}

    this.audioContext = null;
    this.running = false;
    this.paused = false;

    console.warn("[playback] hard stopped");
  }

  pause() {
    if (!this.running || this.paused) return;
    this.paused = true;
    this._emit("PLAYBACK_PAUSED", {});
  }

  resume() {
    if (!this.running || !this.paused) return;
    this.paused = false;
    this._emit("PLAYBACK_RESUMED", {});
    this._pump();
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  enqueueFrame(frame) {
    if (!this.running || this.paused) return;

    const { seqNum, pcmBytes } = frame;

    if (!(pcmBytes instanceof Uint8Array)) {
      throw new Error("enqueueFrame expects pcmBytes as Uint8Array");
    }

    if (pcmBytes.byteLength !== AUDIO_BYTES_PER_FRAME_PCM) {
      this._framesDropped++;
      this._emit("PLAYBACK_FRAME_DROPPED_INVALID_LEN", {
        seqNum,
        expected: AUDIO_BYTES_PER_FRAME_PCM,
        actual: pcmBytes.byteLength,
      });
      return;
    }

    // Sequence gap detection (debug only)
    // Sequence gap detection (debug only)
    if (this._nextSeqExpected == null) {
      // First frame sets expectation for the *next* frame
      this._nextSeqExpected = seqNum + 1;
      if (this._nextSeqExpected > SEQ_NUM_MAX) {
        this._nextSeqExpected = SEQ_NUM_START;
      }
    } else {
      const expected = this._nextSeqExpected;
      if (seqNum !== expected) {
        this._emit("PLAYBACK_SEQ_GAP", { expected, actual: seqNum });
      }
      this._nextSeqExpected = seqNum + 1;
      if (this._nextSeqExpected > SEQ_NUM_MAX) {
        this._nextSeqExpected = SEQ_NUM_START;
      }
    }

    const samples = pcm16BytesToFloat32(pcmBytes);

    if (samples.length !== AUDIO_SAMPLES_PER_FRAME) {
      this._framesDropped++;
      this._emit("PLAYBACK_FRAME_DROPPED_INVALID_SAMPLES", {
        seqNum,
        expected: AUDIO_SAMPLES_PER_FRAME,
        actual: samples.length,
      });
      return;
    }

    // For Phase 5 echo path, queue remains shallow.
    this._queue.push({
      seqNum,
      samples,
      receivedAt: frame.receivedAt ?? null,
    });

    this._framesEnqueued++;

    const now = this.audioContext.currentTime;
    if (this._scheduledUntil < now) {
      this._scheduledUntil = now;
    }

    this._pump();
  }

  async hardStopAndRestart() {
    const wasRunning = this.running;
    this.stop();
    if (wasRunning) {
      await this.start();
    }
  }

  stats() {
    return {
      framesEnqueued: this._framesEnqueued,
      framesScheduled: this._framesScheduled,
      framesDropped: this._framesDropped,
      queueFrames: this._queue.length,
      scheduledAheadMs: this.running
        ? Math.max(
            0,
            (this._scheduledUntil - this.audioContext.currentTime) * 1000
          )
        : 0,
    };
  }

  // ---------------------------------------------------------------------------
  // Internals
  // ---------------------------------------------------------------------------

  _pump() {
    console.log("[playback timing]", {
      now: this.audioContext.currentTime,
      scheduledUntil: this._scheduledUntil,
      queue: this._queue.length,
    });
    if (!this.running || this.paused || !this.audioContext) return;

    const epoch = this._epoch;
    const now = this.audioContext.currentTime;

    // Schedule slightly ahead to avoid underruns while keeping latency low
    const SCHEDULE_AHEAD_S = 0.10;

    if (this._scheduledUntil < now) {
      this._scheduledUntil = now;
    }

    while (
      this._queue.length > 0 &&
      this._scheduledUntil < now + SCHEDULE_AHEAD_S
    ) {
      const item = this._queue.shift();
      if (!item) break;

      let playbackLatencyMs = null;
      if (item.receivedAt != null) {
        playbackLatencyMs = Math.round(
          performance.now() - item.receivedAt
        );
      }

      const buffer = this.audioContext.createBuffer(
        AUDIO_CHANNELS,
        AUDIO_SAMPLES_PER_FRAME,
        AUDIO_SAMPLE_RATE_HZ
      );
      buffer.copyToChannel(item.samples, 0);

      const source = this.audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(this.audioContext.destination);

      const startAt = this._scheduledUntil;

      source.onended = () => {
        // Ignore if stop() was called (epoch changed)
        if (this._epoch !== epoch) return;
        this._pump();
      };

      source.start(startAt);

      this._framesScheduled++;

      this._emit("PLAYBACK_FRAME_SCHEDULED", {
        seqNum: item.seqNum,
        startAtMsFromNow: Math.round((startAt - now) * 1000),
        scheduledAheadMs: Math.round(
          (this._scheduledUntil - now) * 1000
        ),
        playbackLatencyMs,
      });

      this._scheduledUntil += AUDIO_FRAME_DURATION_S;
    }
  }

  _emit(type, fields) {
    if (!this.onEvent) return;
    try {
      this.onEvent({
        type,
        ts: performance.now(),
        ...fields,
      });
    } catch {
      // Never let observability throw in audio hot path
    }
  }
}
