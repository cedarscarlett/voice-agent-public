/**
 * frontend/audio/audio_format.js
 *
 * Audio format constants (SPEC MIRROR).
 *
 * Rules:
 * - Mirrors backend/spec.py exactly (audio + binary framing invariants)
 * - No magic numbers elsewhere in frontend audio code
 * - If changing audio behavior, update BOTH backend/spec.py and this file
 */

// -----------------------------------------------------------------------------
// PCM format (v1)
// -----------------------------------------------------------------------------

export const AUDIO_SAMPLE_RATE_HZ = 16000;
export const AUDIO_CHANNELS = 1;
export const AUDIO_SAMPLE_WIDTH_BYTES = 2; // PCM16
export const AUDIO_FRAME_MS = 20;

// Derived values
export const AUDIO_SAMPLES_PER_FRAME =
  (AUDIO_SAMPLE_RATE_HZ * AUDIO_FRAME_MS) / 1000;

export const AUDIO_BYTES_PER_FRAME_PCM =
  AUDIO_SAMPLES_PER_FRAME * AUDIO_SAMPLE_WIDTH_BYTES;

export const AUDIO_FRAME_DURATION_S = AUDIO_FRAME_MS / 1000.0;

// -----------------------------------------------------------------------------
// Browser capture defaults
// -----------------------------------------------------------------------------

// Most browsers capture at 48kHz by default.
// We resample client-side to 16kHz before sending.
export const BROWSER_CAPTURE_SAMPLE_RATE_HZ_DEFAULT = 48000;

// -----------------------------------------------------------------------------
// Binary protocol framing (client → server)
// -----------------------------------------------------------------------------

// Client → Server (mic audio)
// [4 bytes seq_num][640 bytes PCM]
export const C2S_SEQ_NUM_BYTES = 4;
export const C2S_FRAME_BYTES_TOTAL =
  C2S_SEQ_NUM_BYTES + AUDIO_BYTES_PER_FRAME_PCM;

// -----------------------------------------------------------------------------
// Binary protocol framing (server → client)
// -----------------------------------------------------------------------------

// Server → Client (TTS audio)
// [4 bytes seq_num][4 bytes run_id][640 bytes PCM]
export const S2C_SEQ_NUM_BYTES = 4;
export const S2C_RUN_ID_BYTES = 4;
export const S2C_FRAME_BYTES_TOTAL =
  S2C_SEQ_NUM_BYTES + S2C_RUN_ID_BYTES + AUDIO_BYTES_PER_FRAME_PCM;

// -----------------------------------------------------------------------------
// Sequence numbers
// -----------------------------------------------------------------------------

export const SEQ_NUM_START = 1;
export const SEQ_NUM_MAX = 2 ** 32 - 1;

// -----------------------------------------------------------------------------
// Barge-in detection (spec §8)
// -----------------------------------------------------------------------------

export const BARGE_IN_HOLD_MS_MIN = 80;
export const BARGE_IN_HOLD_MS_MAX = 150;
export const BARGE_IN_RMS_THRESHOLD_DEFAULT = 0.08;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/**
 * Convert number of audio frames to seconds.
 * Canonical depth unit (spec §10).
 */
export function framesToSeconds(numFrames) {
  if (numFrames <= 0) return 0;
  return numFrames * AUDIO_FRAME_DURATION_S;
}

/**
 * Convert duration in seconds to number of frames (floor).
 * Canonical invariant (mirrors backend/spec.py).
 */
export function secondsToFrames(durationSeconds) {
  if (durationSeconds <= 0) return 0;
  return Math.floor(durationSeconds / AUDIO_FRAME_DURATION_S);
}

/**
 * Validate PCM frame byte length.
 * Defensive check before sending over wire.
 */
export function validatePcmFrameLength(byteLength) {
  return byteLength === AUDIO_BYTES_PER_FRAME_PCM;
}

