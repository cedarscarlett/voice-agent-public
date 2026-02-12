/**
 * frontend/audio/barge_in.js
 *
 * Frontend-only barge-in detection (Phase 5).
 *
 * Purpose:
 * - Detect user speech while TTS audio is playing
 * - Immediately stop playback (caller responsibility)
 * - Emit a BARGE_IN signal to higher layers (WS client / orchestrator)
 *
 * Design constraints (spec §8):
 * - Frontend-only detection (no server VAD)
 * - Simple, deterministic signal (RMS threshold)
 * - Hold-time based to avoid false positives
 * - No buffering, no DSP complexity
 *
 * Detection behavior:
 * - When enabled and signal is held above threshold for holdMs:
 *   - Fires onBargeIn() callback once
 *   - Auto-disables to prevent repeated firing
 *   - Caller must explicitly re-enable() for the next detection cycle
 *
 * This one-shot behavior prevents double-triggering during a single
 * speech event.
 *
 * Tuning guidance:
 *
 * - rmsThreshold:
 *   - ~0.08 works for typical desk microphones
 *   - Too low (< 0.05): false positives from room noise
 *   - Too high (> 0.15): misses soft speech
 *
 * - holdMs:
 *   - 80–100ms recommended
 *   - Too short (< 50ms): triggers on mouth noise / clicks
 *   - Too long (> 200ms): feels laggy; user must speak longer
 *
 * Test in:
 * - quiet room
 * - noisy room
 * - different mic distances
 *
 * This module:
 * - Does NOT stop playback itself
 * - Does NOT send WebSocket messages
 * - Only emits a signal when barge-in is detected
 *
 * Usage:
 *
 *   const bargeIn = new BargeInDetector({
 *     onBargeIn: () => {
 *       playback.hardStopAndRestart();
 *       ws.sendJson("BARGE_IN");
 *     }
 *   });
 *
 *   await bargeIn.start(audioContext);
 *
 *   // While SPEAKING:
 *   bargeIn.enable();
 *
 *   // Otherwise:
 *   bargeIn.disable();
 */

import {
  BARGE_IN_HOLD_MS_MIN,
  BARGE_IN_HOLD_MS_MAX,
  BARGE_IN_RMS_THRESHOLD_DEFAULT,
} from "./audio_format.js";

// Poll cadence for RMS detection.
// Aligned with 20ms audio frame timing.
// Phase 5: constant is sufficient; can be made configurable later.
const BARGE_IN_POLL_INTERVAL_MS = 20;

export class BargeInDetector {
  /**
   * @param {object} opts
   * @param {function():void} opts.onBargeIn - callback fired once per detection
   * @param {number} [opts.rmsThreshold] - override default RMS threshold
   * @param {number} [opts.holdMs] - override hold duration (clamped)
   */
  constructor({ onBargeIn, rmsThreshold, holdMs }) {
    if (typeof onBargeIn !== "function") {
      throw new Error("BargeInDetector requires onBargeIn callback");
    }

    this.onBargeIn = onBargeIn;

    this.rmsThreshold =
      rmsThreshold != null
        ? rmsThreshold
        : BARGE_IN_RMS_THRESHOLD_DEFAULT;

    // Clamp hold duration to spec range
    const desiredHold =
      holdMs != null ? holdMs : BARGE_IN_HOLD_MS_MIN;
    this.holdMs = Math.min(
      Math.max(desiredHold, BARGE_IN_HOLD_MS_MIN),
      BARGE_IN_HOLD_MS_MAX
    );

    this.audioContext = null;
    this.analyser = null;
    this.micSource = null;

    // MediaStream must be retained so tracks can be stopped on cleanup.
    // Otherwise the browser mic indicator remains active.
    this.mediaStream = null;

    this.enabled = false;
    this._running = false;

    this._aboveThresholdSinceMs = null;
    this._pollInterval = null;

    // Reusable buffer to avoid allocations
    this._timeDomainBuffer = null;
  }

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  /**
   * Start microphone monitoring.
   *
   * Requires an existing AudioContext.
   * Uses a lightweight AnalyserNode for RMS calculation.
   */
  async start(audioContext) {
    if (this._running) return;
    if (!audioContext) {
      throw new Error("BargeInDetector.start requires an AudioContext");
    }

    this.audioContext = audioContext;

    // getUserMedia calls. Some browsers / OSes struggle with concurrent
    // microphone access or show multiple permission prompts.
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,

          // Keep raw signal for predictable RMS thresholding.
          // Note: Disabling AGC means thresholds may need tuning
          // per-device / environment.
          autoGainControl: false,
        },
      });
    } catch (err) {
      console.error("[barge-in] getUserMedia failed", err);
      throw err;
    }

    this.micSource =
      this.audioContext.createMediaStreamSource(this.mediaStream);

    this.analyser = this.audioContext.createAnalyser();

    // fftSize tradeoff:
    // 2048 samples ≈ ~43ms window @ 48kHz.
    // Lower values (512–1024) reduce latency and are sufficient for RMS,
    // but 2048 is conservative and stable for Phase 5.
    this.analyser.fftSize = 2048;

    // No smoothing: we want instant signal changes.
    // Hold-time (80–150ms) provides noise rejection.
    this.analyser.smoothingTimeConstant = 0.0;

    this.micSource.connect(this.analyser);

    this._timeDomainBuffer = new Float32Array(this.analyser.fftSize);

    this._pollInterval = setInterval(
      () => this._poll(),
      BARGE_IN_POLL_INTERVAL_MS
    );

    this._running = true;
  }

  /**
   * Stop microphone monitoring and release resources.
   */
  stop() {
    if (!this._running) return;

    if (this._pollInterval) {
      clearInterval(this._pollInterval);
      this._pollInterval = null;
    }

    try {
      if (this.micSource) {
        this.micSource.disconnect();
      }
      if (this.analyser) {
        this.analyser.disconnect();
      }
    } catch {
      // no-op
    }

    // IMPORTANT: Release microphone tracks.
    // Failing to do this leaves the browser mic indicator active
    // and leaks device resources.
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    this.audioContext = null;
    this.analyser = null;
    this.micSource = null;

    this._timeDomainBuffer = null;
    this._aboveThresholdSinceMs = null;
    this._running = false;
  }

  // ---------------------------------------------------------------------------
  // Control
  // ---------------------------------------------------------------------------

  /**
   * Enable barge-in detection.
   * Typically called when entering SPEAKING state.
   */
  enable() {
    this.enabled = true;
    this._aboveThresholdSinceMs = null;
  }

  /**
   * Disable barge-in detection.
   * Typically called when leaving SPEAKING state.
   */
  disable() {
    this.enabled = false;
    this._aboveThresholdSinceMs = null;
  }

  // ---------------------------------------------------------------------------
  // Detection logic
  // ---------------------------------------------------------------------------

  _poll() {
    if (!this.enabled || !this.analyser) return;

    this.analyser.getFloatTimeDomainData(this._timeDomainBuffer);

    // Use RMS (not peak) for more stable detection:
    // - Peak is overly sensitive to transient clicks/pops
    // - RMS averages energy over the window, reducing false positives
    let sumSquares = 0.0;
    for (let i = 0; i < this._timeDomainBuffer.length; i++) {
      const v = this._timeDomainBuffer[i];
      sumSquares += v * v;
    }

    const rms = Math.sqrt(sumSquares / this._timeDomainBuffer.length);
    const nowMs = performance.now();

    if (rms >= this.rmsThreshold) {
      if (this._aboveThresholdSinceMs == null) {
        this._aboveThresholdSinceMs = nowMs;
      } else {
        const heldMs = nowMs - this._aboveThresholdSinceMs;
        if (heldMs >= this.holdMs) {
          // Barge-in detected (one-shot)
          this._aboveThresholdSinceMs = null;
          this.enabled = false;

          try {
            this.onBargeIn();
          } catch (err) {
            console.error("[barge-in] onBargeIn callback failed", err);
          }
        }
      }
    } else {
      // Reset if signal drops below threshold
      this._aboveThresholdSinceMs = null;
    }
  }
}
