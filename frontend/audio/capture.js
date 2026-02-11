/**
 * frontend/audio/capture.js
 *
 * Microphone capture + resampling to PCM16 @ 16kHz.
 *
 * Phase 5 responsibilities:
 * - Capture mic audio
 * - Resample client-side to 16kHz
 * - Frame into exact 20ms PCM16 buffers
 * - Generate sequence numbers at the source
 * - Emit capture timestamps for latency measurement
 * - Expose pause/resume hooks for backpressure
 *
 * NOT responsible for:
 * - WebSocket sending
 * - Barge-in logic
 * - Any orchestration decisions
 */

import {
  AUDIO_SAMPLE_RATE_HZ,
  AUDIO_FRAME_MS,
  AUDIO_SAMPLES_PER_FRAME,
  AUDIO_BYTES_PER_FRAME_PCM,
  BROWSER_CAPTURE_SAMPLE_RATE_HZ_DEFAULT,
  SEQ_NUM_START,
  SEQ_NUM_MAX,
} from "./audio_format.js";

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

function floatToPCM16(float32Array) {
  const pcm16 = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm16;
}

// Simple linear resampler (good enough for Phase 5)
//
// Production note:
// For 48kHz → 16kHz, linear interpolation introduces mild aliasing.
// For voice at 16kHz target, this is acceptable.
// If audio quality issues arise, consider proper low-pass filtering + decimation.
function resampleLinear(input, inputRate, outputRate) {
  if (inputRate === outputRate) return input;

  const ratio = inputRate / outputRate;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i++) {
    const srcIndex = i * ratio;
    const i0 = Math.floor(srcIndex);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = srcIndex - i0;
    output[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }

  return output;
}

// -----------------------------------------------------------------------------
// MicCapture
// -----------------------------------------------------------------------------

export class MicCapture {
  constructor({ onFrame }) {
    /**
     * onFrame({
     *   seqNum: number,
     *   pcmBytes: Uint8Array,
     *   capturedAt: number (performance.now)
     * })
     */
    this.onFrame = onFrame;

    this.audioContext = null;
    this.mediaStream = null;
    this.sourceNode = null;
    this.processorNode = null;

    this.inputSampleRate = null;
    this.running = false;
    this.paused = false;

    // True ring buffer (1 second capacity = 16,000 samples @ 16kHz)
    // Uses modulo arithmetic for wrapping, preventing overflow
    this._buffer = new Float32Array(AUDIO_SAMPLE_RATE_HZ);
    this._writePos = 0;
    this._readPos = 0;

    this._nextSeqNum = SEQ_NUM_START;
  }

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  async start() {
    if (this.running) return;

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch (err) {
      console.error("[mic] getUserMedia failed", err);
      throw new Error(`Mic access denied: ${err.message}`);
    }

    // ScriptProcessor is deprecated but intentionally used here:
    // - AudioWorklet requires extra files + worker plumbing (overkill for Phase 5)
    // - ScriptProcessor timing is predictable and debuggable
    // - Chrome support extends well past 2025
    //
    // TODO(Phase 6+): Migrate to AudioWorklet for production.
    this.audioContext = new AudioContext({
      sampleRate: BROWSER_CAPTURE_SAMPLE_RATE_HZ_DEFAULT,
    });

    this.inputSampleRate = this.audioContext.sampleRate;

    if (this.inputSampleRate !== BROWSER_CAPTURE_SAMPLE_RATE_HZ_DEFAULT) {
      console.warn("[mic] unexpected capture sample rate", {
        requested: BROWSER_CAPTURE_SAMPLE_RATE_HZ_DEFAULT,
        actual: this.inputSampleRate,
      });
    }

    this.sourceNode = this.audioContext.createMediaStreamSource(
      this.mediaStream
    );

    this.processorNode = this.audioContext.createScriptProcessor(0, 1, 1);

    this.processorNode.onaudioprocess = (event) => {
      if (!this.running || this.paused) return;

      const buffer = event.inputBuffer;

      // 1. No buffer at all
      if (!buffer) return;

      // 2. No frames
      if (buffer.length === 0) return;

      // 3. No channels or empty channel
      if (buffer.numberOfChannels === 0) return;

      const input = buffer.getChannelData(0);

      // 4. TypedArray can exist but be length 0
      if (!input || input.length === 0) return;

      this._handleInput(input);
    };

    this.sourceNode.connect(this.processorNode);
    this.processorNode.connect(this.audioContext.destination);

    this.running = true;

    console.info("[mic] started", {
      inputSampleRate: this.inputSampleRate,
      targetSampleRate: AUDIO_SAMPLE_RATE_HZ,
      frameMs: AUDIO_FRAME_MS,
    });
  }

  stop() {
    if (!this.running) return;

    this.running = false;

    try {
      this.processorNode.disconnect();
      this.sourceNode.disconnect();
    } catch {}

    this.mediaStream.getTracks().forEach((t) => t.stop());
    this.audioContext.close();

    this._writePos = 0;
    this._readPos = 0;

    console.info("[mic] stopped");
  }

  pause() {
    if (this.paused) return;

    this.paused = true;
    console.warn("[mic] paused (backpressure)", {
      bufferedSamples: this._writePos - this._readPos,
      timestamp: performance.now(),
    });
  }

  resume() {
    if (!this.paused) return;

    this.paused = false;
    console.warn("[mic] resumed", {
      timestamp: performance.now(),
    });
  }

  // ---------------------------------------------------------------------------
  // Audio handling
  // ---------------------------------------------------------------------------

  _handleInput(float32Chunk) {
    if (!float32Chunk || float32Chunk.length === 0) {
      return;
    }

    const resampled = resampleLinear(
      float32Chunk,
      this.inputSampleRate,
      AUDIO_SAMPLE_RATE_HZ
    );

    const bufferLen = this._buffer.length;
    const buffered = this._writePos - this._readPos;

    // CRITICAL: Detect buffer overrun BEFORE writing
    if (buffered + resampled.length > bufferLen) {
      console.error("[mic] BUFFER OVERRUN IMMINENT", {
        buffered,
        incoming: resampled.length,
        capacity: bufferLen,
        wouldOverrunBy: buffered + resampled.length - bufferLen,
        timestamp: performance.now(),
      });

      // Trigger backpressure (you have this mechanism already)
      this.pause();

      // Drop this chunk to prevent corruption
      // Production: You might queue this in a secondary overflow buffer instead
      return;
    }

    // Write with wrapping (optimized: compute modulo once, use set())
    const writeStart = this._writePos % bufferLen;
    const writeEnd = writeStart + resampled.length;

    if (writeEnd <= bufferLen) {
      // No wrap: single typed array copy (fast path)
      this._buffer.set(resampled, writeStart);
    } else {
      // Wraps: two copies
      const firstChunkLen = bufferLen - writeStart;
      this._buffer.set(resampled.subarray(0, firstChunkLen), writeStart);
      this._buffer.set(resampled.subarray(firstChunkLen), 0);
    }

    this._writePos += resampled.length;

    // Emit exact 20ms frames
    while (this._writePos - this._readPos >= AUDIO_SAMPLES_PER_FRAME) {
      const readStart = this._readPos % bufferLen;
      const readEnd = readStart + AUDIO_SAMPLES_PER_FRAME;

      let frameSamples;

      if (readEnd <= bufferLen) {
        // No wrap: direct slice (zero-copy view)
        frameSamples = this._buffer.subarray(readStart, readEnd);
      } else {
        // Wraps: manual copy
        frameSamples = new Float32Array(AUDIO_SAMPLES_PER_FRAME);
        const firstChunkLen = bufferLen - readStart;
        frameSamples.set(this._buffer.subarray(readStart, bufferLen), 0);
        frameSamples.set(
          this._buffer.subarray(0, readEnd - bufferLen),
          firstChunkLen
        );
      }

      this._readPos += AUDIO_SAMPLES_PER_FRAME;

      const pcm16 = floatToPCM16(frameSamples);
      const pcmBytes = new Uint8Array(pcm16.buffer);

      // This check should now NEVER trigger
      if (pcmBytes.byteLength !== AUDIO_BYTES_PER_FRAME_PCM) {
        console.error("[mic] INVALID FRAME (IMPOSSIBLE)", {
          byteLength: pcmBytes.byteLength,
          expected: AUDIO_BYTES_PER_FRAME_PCM,
          readStart,
          readEnd,
          bufferLen,
          writePos: this._writePos,
          readPos: this._readPos,
          stack: new Error().stack,
        });
        continue;
      }

      const seqNum = this._nextSeqNum;
      const capturedAt = performance.now();

      this.onFrame({
        seqNum,
        pcmBytes,
        capturedAt,
      });

      // Periodic debug logging - Exactly 1s (50 frames × 20ms)
      if (seqNum % 50 === 0) {
        console.debug("[mic] emitted frame", {
          seqNum,
          byteLength: pcmBytes.byteLength,
          buffered: this._writePos - this._readPos,
          writePos: this._writePos,
          readPos: this._readPos,
        });
      }

      this._nextSeqNum++;
      if (this._nextSeqNum > SEQ_NUM_MAX) {
        this._nextSeqNum = SEQ_NUM_START;
      }
    }

    // Optional: Prevent unbounded position growth
    // Wrap positions when both are aligned to buffer boundary
    // This keeps numbers bounded while preserving (writePos - readPos) invariant
    if (this._readPos >= bufferLen && this._readPos % bufferLen === 0) {
      const offset = Math.floor(this._readPos / bufferLen) * bufferLen;
      this._readPos -= offset;
      this._writePos -= offset;

      console.debug("[mic] normalized positions", {
        newReadPos: this._readPos,
        newWritePos: this._writePos,
        buffered: this._writePos - this._readPos,
      });
    }
  }
}