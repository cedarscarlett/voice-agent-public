/**
 * frontend/ws/client.js
 *
 * Single-WebSocket client for:
 * - JSON control messages (text frames)
 * - Binary audio frames:
 *   - C→S mic: 4B seq_num (u32 LE) + 640B PCM16
 *   - S→C tts: 4B seq_num (u32 LE) + 4B run_id (u32 LE) + 640B PCM16
 *
 * Phase 5 responsibilities:
 * - Connect/reconnect with simple backoff
 * - Send MIC frames with seq numbers (from capture.js)
 * - Receive TTS frames, drop stale run_id, forward to playback.js
 * - Apply basic send-side backpressure signaling (pause/resume mic capture)
 * - Emit observability-friendly events (no orchestration logic)
 *
 * NOT responsible for:
 * - UI rendering (callers provide callbacks)
 * - Orchestrator decisions / state machine (backend-owned)
 */

import {
  AUDIO_BYTES_PER_FRAME_PCM,
  C2S_SEQ_NUM_BYTES,
  C2S_FRAME_BYTES_TOTAL,
  S2C_SEQ_NUM_BYTES,
  S2C_RUN_ID_BYTES,
  S2C_FRAME_BYTES_TOTAL,
  SEQ_NUM_START,
  SEQ_NUM_MAX,
} from "../audio/audio_format.js";

// -----------------------------------------------------------------------------
// Binary helpers (u32 little-endian)
// -----------------------------------------------------------------------------

function _readU32LE(view, offset) {
  return view.getUint32(offset, true);
}

function _writeU32LE(view, offset, value) {
  view.setUint32(offset, value >>> 0, true);
}

// -----------------------------------------------------------------------------
// Defaults
// -----------------------------------------------------------------------------

const DEFAULT_RETRY_BACKOFF_MS = [200, 400, 800];

// ~500ms worth of mic frames buffered in the WS send queue (25 frames @ 20ms)
const DEFAULT_MIC_BACKPRESSURE_THRESHOLD_BYTES = C2S_FRAME_BYTES_TOTAL * 25;
// Resume once drained under ~200ms (10 frames)
const DEFAULT_MIC_BACKPRESSURE_RESUME_BYTES = C2S_FRAME_BYTES_TOTAL * 10;

// -----------------------------------------------------------------------------
// WS Client
// -----------------------------------------------------------------------------

export class WSClient {
  /**
   * @param {object} opts
   * @param {string} opts.url - ws://... endpoint
   * @param {function(object):void} [opts.onJson] - receives parsed JSON messages
   * @param {function(object):void} [opts.onEvent] - observability hook
   * @param {function(string):void} [opts.onStatus] - "CONNECTING" | "UP" | "DOWN"
   * @param {import("../audio/playback.js").Playback} [opts.playback] - playback instance
   * @param {object} [opts.micControl] - { pause():void, resume():void } (optional)
   * @param {number[]} [opts.retryBackoffMs] - reconnect backoff sequence
   * @param {number} [opts.micBackpressureThresholdBytes]
   * @param {number} [opts.micBackpressureResumeBytes]
   */
  constructor(opts) {
    this.url = opts.url;

    this.onJson = opts.onJson || null;
    this.onEvent = opts.onEvent || null;
    this.onStatus = opts.onStatus || null;

    this.playback = opts.playback || null;
    this.micControl = opts.micControl || null;

    this.retryBackoffMs = opts.retryBackoffMs || DEFAULT_RETRY_BACKOFF_MS;

    this.micBackpressureThresholdBytes =
      opts.micBackpressureThresholdBytes ||
      DEFAULT_MIC_BACKPRESSURE_THRESHOLD_BYTES;

    this.micBackpressureResumeBytes =
      opts.micBackpressureResumeBytes || DEFAULT_MIC_BACKPRESSURE_RESUME_BYTES;

    this.ws = null;

    this.connectionStatus = "DOWN"; // "CONNECTING" | "UP" | "DOWN"
    this._retryAttempt = 0;
    this._closedByUser = false;

    // Client-side stale filtering for TTS frames
    this.activeTtsRunId = null; // u32 or null

    // Sequence gap debug (TTS side)
    this._lastTtsSeq = null;

    // Backpressure gating
    this._micPausedForBackpressure = false;

    // Timer for periodic WS bufferedAmount checks
    this._bpInterval = null;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  setPlayback(playback) {
    this.playback = playback;
  }

  setMicControl(micControl) {
    this.micControl = micControl;
  }

  setActiveTtsRunId(runId) {
    // run_id starts at 1 on the wire; null means "accept any until set"
    this.activeTtsRunId = runId;
    this._emit("CLIENT_ACTIVE_TTS_RUN_SET", { runId });
  }

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this._closedByUser = false;
    this._setStatus("CONNECTING");

    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    ws.onopen = () => {
      this._retryAttempt = 0;
      this._setStatus("UP");
      this._emit("WS_OPEN", { url: this.url });

      // Start backpressure monitor loop while connected
      this._startBackpressureMonitor();
    };

    ws.onmessage = (evt) => {
      if (evt.data instanceof ArrayBuffer) {
        const buf = evt.data;
        console.log(
          "[RX]",
          "byteLength=", buf.byteLength,
          "seqLE=", new DataView(buf).getUint32(0, true),
          "runLE=", new DataView(buf).getUint32(4, true),
          "activeTtsRunId=", this.activeTtsRunId,
        );
      }
      if (typeof evt.data === "string") {
        this._handleText(evt.data);
      } else {
        this._handleBinary(evt.data);
      }
    };

    ws.onerror = (err) => {
      // Browser gives limited info; treat as signal
      this._emit("WS_ERROR", { message: String(err) });
    };

    ws.onclose = (evt) => {
      this._emit("WS_CLOSE", {
        code: evt.code,
        reason: evt.reason,
        wasClean: evt.wasClean,
      });

      this._stopBackpressureMonitor();
      this._setStatus("DOWN");
      this.ws = null;



      if (!this._closedByUser) {
        this._scheduleReconnect();
      }
    };
  }

  disconnect() {
    this._closedByUser = true;
    this._stopBackpressureMonitor();

    if (this.ws) {
      try {
        this.ws.close(1000, "client_disconnect");
      } catch {}
    }

    this.ws = null;
    this._setStatus("DOWN");
  }

  /**
   * Send JSON message: { type, ts_ms, ...fields }
   */
  sendJson(type, fields = {}) {
    if (!this._isOpen()) return false;

    const msg = {
      type,
      ts_ms: Date.now(),
      ...fields,
    };

    try {
      this.ws.send(JSON.stringify(msg));
      this._emit("JSON_SENT", { type });
      return true;
    } catch (err) {
      this._emit("JSON_SEND_FAILED", { type, error: String(err) });
      return false;
    }
  }

  /**
   * Send one mic audio frame.
   *
   * @param {object} frame
   * @param {number} frame.seqNum - u32 seq, starts at 1, wraps at SEQ_NUM_MAX
   * @param {Uint8Array} frame.pcmBytes - exactly 640 bytes
   * @param {number} [frame.capturedAt] - perf timestamp (from capture.js)
   */
  sendMicFrame(frame) {
    if (!this._isOpen()) return false;

    const { seqNum, pcmBytes } = frame;

    if (!(pcmBytes instanceof Uint8Array)) {
      throw new Error("sendMicFrame expects pcmBytes as Uint8Array");
    }
    if (pcmBytes.byteLength !== AUDIO_BYTES_PER_FRAME_PCM) {
      this._emit("MIC_FRAME_REJECTED_INVALID_LEN", {
        seqNum,
        expected: AUDIO_BYTES_PER_FRAME_PCM,
        actual: pcmBytes.byteLength,
      });
      return false;
    }
    if (seqNum < SEQ_NUM_START || seqNum > SEQ_NUM_MAX) {
      this._emit("MIC_FRAME_REJECTED_INVALID_SEQ", { seqNum });
      return false;
    }

    // Backpressure: if WS is already too backed up, pause capture.
    // We still attempt send; but if bufferedAmount is huge, the browser will queue.
    this._applyBackpressureIfNeeded();

    const buf = new ArrayBuffer(C2S_FRAME_BYTES_TOTAL);
    const view = new DataView(buf);

    _writeU32LE(view, 0, seqNum);

    // Copy PCM bytes after header
    const outBytes = new Uint8Array(buf);
    outBytes.set(pcmBytes, C2S_SEQ_NUM_BYTES);

    try {
      this.ws.send(buf);

      this._emit("MIC_FRAME_SENT", {
        seqNum,
        bufferedAmount: this.ws.bufferedAmount,
        capturedAt: frame.capturedAt ?? null,
      });

      // Backpressure: re-check after enqueue
      this._applyBackpressureIfNeeded();
      return true;
    } catch (err) {
      this._emit("MIC_FRAME_SEND_FAILED", { seqNum, error: String(err) });
      return false;
    }
  }

  // ---------------------------------------------------------------------------
  // Text/Binary handlers
  // ---------------------------------------------------------------------------

  _handleText(text) {
    let data;
    try {
      data = JSON.parse(text);
    } catch (err) {
      this._emit("JSON_PARSE_ERROR", {
        error: String(err),
        preview: text.slice(0, 120),
      });
      return;
    }

    // ------------------------------------------------------------
    // HARD STOP: server-directed audio cancellation (barge-in)
    // ------------------------------------------------------------
    if (data && data.type === "AUDIO_STOP") {
      const { service, run_id } = data;

      this._emit("AUDIO_STOP_RECEIVED", { service, run_id });

      if (service === "TTS" && this.playback) {
        try {
          // Immediate silence: kill AudioContext + scheduled buffers
          this.playback.stop();
        } catch (err) {
          this._emit("AUDIO_STOP_PLAYBACK_FAILED", {
            error: String(err),
          });
        }
      }

      // Sentinel: reject ALL TTS frames until an explicit TTS_START arrives
      this.activeTtsRunId =-1;

      // Do NOT forward to app-level onJson
      return;
    }

    // ------------------------------------------------------------
    // TTS_START: declare the only valid TTS run
    // ------------------------------------------------------------
    if (data?.type === "TTS_START") {
      const { run_id } = data;

      this._emit("TTS_START_RECEIVED", { run_id });

      // Accept frames ONLY for this run
      this.activeTtsRunId = run_id;

      // Restart playback if it was previously hard-stopped
      if (this.playback && !this.playback.running) {
        try {
          this.playback.start();
        } catch (err) {
          this._emit("PLAYBACK_RESTART_FAILED", { error: String(err) });
        }
      }

      // Still forward to app-level (UI may care)
      if (this.onJson) {
        try {
          this.onJson(data);
        } catch (err) {
          this._emit("ONJSON_HANDLER_ERROR", { error: String(err) });
        }
      }
      return;
    }


    // ------------------------------------------------------------
    // SESSION_INIT (unchanged)
    // ------------------------------------------------------------
    if (data && data.type === "SESSION_INIT") {
      this._emit("SESSION_INIT", { session_id: data.session_id ?? null });
    }

    if (this.onJson) {
      try {
        this.onJson(data);
      } catch (err) {
        this._emit("ONJSON_HANDLER_ERROR", { error: String(err) });
      }
    }
  }

  _handleBinary(payload) {
    const receivedAt = performance.now();

    if (!(payload instanceof ArrayBuffer)) {
      // Some browsers may deliver Blob; handle defensively
      if (payload instanceof Blob) {
        payload.arrayBuffer().then((buf) => this._handleBinary(buf));
        return;
      }
      this._emit("BINARY_UNSUPPORTED_PAYLOAD", { type: typeof payload });
      return;
    }

    if (payload.byteLength !== S2C_FRAME_BYTES_TOTAL) {
      this._emit("TTS_FRAME_REJECTED_INVALID_LEN", {
        expected: S2C_FRAME_BYTES_TOTAL,
        actual: payload.byteLength,
      });
      return;
    }

    const view = new DataView(payload);
    const seqNum = _readU32LE(view, 0);
    const runId = _readU32LE(view, S2C_SEQ_NUM_BYTES);

    if (seqNum < SEQ_NUM_START || seqNum > SEQ_NUM_MAX) {
      this._emit("TTS_FRAME_REJECTED_INVALID_SEQ", { seqNum, runId });
      return;
    }
    if (runId < 1) {
      this._emit("TTS_FRAME_REJECTED_INVALID_RUN", { seqNum, runId });
      return;
    }

    // Drop stale runs (belt + suspenders with backend)
    if (this.activeTtsRunId != null && runId !== this.activeTtsRunId) {
      this._emit("TTS_FRAME_DROPPED_STALE_RUN", {
        seqNum,
        runId,
        activeTtsRunId: this.activeTtsRunId,
      });
      return;
    }

    // Sequence gap debug (TTS)
    if (this._lastTtsSeq == null) {
      this._lastTtsSeq = seqNum;
    } else {
      const expected = this._lastTtsSeq + 1 > SEQ_NUM_MAX ? SEQ_NUM_START : this._lastTtsSeq + 1;
      if (seqNum !== expected) {
        this._emit("TTS_SEQ_GAP", { expected, actual: seqNum, runId });
      }
      this._lastTtsSeq = seqNum;
    }

    // Extract PCM bytes slice (no copy of underlying buffer beyond view)
    const pcmBytes = new Uint8Array(payload, S2C_SEQ_NUM_BYTES + S2C_RUN_ID_BYTES, AUDIO_BYTES_PER_FRAME_PCM);

    // Forward to playback
    if (!this.playback) {
      this._emit("TTS_FRAME_DROPPED_NO_PLAYBACK", { seqNum, runId });
      return;
    }

    try {
      this.playback.enqueueFrame({
        seqNum,
        pcmBytes,
        receivedAt, // used by playback.js to emit playbackLatencyMs
        runId,
      });

      this._emit("TTS_FRAME_RECEIVED", {
        seqNum,
        runId,
        bufferedAmount: this.ws ? this.ws.bufferedAmount : null,
      });
    } catch (err) {
      this._emit("TTS_PLAYBACK_ENQUEUE_FAILED", {
        seqNum,
        runId,
        error: String(err),
      });
    }
  }

  // ---------------------------------------------------------------------------
  // Backpressure
  // ---------------------------------------------------------------------------

  _startBackpressureMonitor() {
    if (this._bpInterval) return;

    // Check ~10x/sec; cheap + good enough
    this._bpInterval = setInterval(() => {
      if (!this._isOpen()) return;
      this._applyBackpressureIfNeeded();
    }, 100);
  }

  _stopBackpressureMonitor() {
    if (!this._bpInterval) return;
    clearInterval(this._bpInterval);
    this._bpInterval = null;
  }

  _applyBackpressureIfNeeded() {
    if (!this._isOpen()) return;

    const buffered = this.ws.bufferedAmount;

    if (!this._micPausedForBackpressure && buffered >= this.micBackpressureThresholdBytes) {
      this._pauseMic("ws_buffer_high", buffered);
    } else if (this._micPausedForBackpressure && buffered <= this.micBackpressureResumeBytes) {
      this._resumeMic("ws_buffer_drained", buffered);
    }
  }

  _pauseMic(reason, bufferedAmount = null) {
    this._micPausedForBackpressure = true;

    this._emit("MIC_BACKPRESSURE_PAUSE", {
      reason,
      bufferedAmount,
      thresholdBytes: this.micBackpressureThresholdBytes,
    });

    if (this.micControl && typeof this.micControl.pause === "function") {
      try {
        this.micControl.pause();
      } catch (err) {
        this._emit("MIC_PAUSE_FAILED", { error: String(err) });
      }
    }
  }

  _resumeMic(reason, bufferedAmount = null) {
    this._micPausedForBackpressure = false;

    this._emit("MIC_BACKPRESSURE_RESUME", {
      reason,
      bufferedAmount,
      resumeBytes: this.micBackpressureResumeBytes,
    });

    if (this.micControl && typeof this.micControl.resume === "function") {
      try {
        this.micControl.resume();
      } catch (err) {
        this._emit("MIC_RESUME_FAILED", { error: String(err) });
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Reconnect
  // ---------------------------------------------------------------------------

  _scheduleReconnect() {
    const delay =
      this.retryBackoffMs[Math.min(this._retryAttempt, this.retryBackoffMs.length - 1)];

    this._emit("WS_RECONNECT_SCHEDULED", {
      attempt: this._retryAttempt + 1,
      delayMs: delay,
    });

    this._retryAttempt++;

    setTimeout(() => {
      if (this._closedByUser) return;
      this.connect();
    }, delay);
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  _isOpen() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  _setStatus(status) {
    if (this.connectionStatus === status) return;
    this.connectionStatus = status;

    if (this.onStatus) {
      try {
        this.onStatus(status);
      } catch (err) {
        this._emit("ONSTATUS_HANDLER_ERROR", { error: String(err) });
      }
    }

    this._emit("WS_STATUS", { status });
  }

  _emit(type, fields = {}) {
    if (!this.onEvent) return;
    try {
      this.onEvent({
        type,
        ts: performance.now(),
        ...fields,
      });
    } catch {
      // never throw from observability
    }
  }
}
