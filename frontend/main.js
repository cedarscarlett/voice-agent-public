import { MicCapture } from "./audio/capture.js";
import { Playback } from "./audio/playback.js";
import { BargeInDetector } from "./audio/barge_in.js";
import { WSClient } from "./ws/client.js";

// -----------------------------------------------------------------------------
// Minimal logging
// -----------------------------------------------------------------------------

const logEl = document.getElementById("log");

function log(...args) {
  console.log(...args);
  if (logEl) {
    logEl.textContent += args.join(" ") + "\n";
  }
}

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------

let mic = null;
let playback = null;
let bargeIn = null;
let ws = null;

const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");

// -----------------------------------------------------------------------------
// Start
// -----------------------------------------------------------------------------

startBtn.onclick = async () => {
  startBtn.disabled = true;
  stopBtn.disabled = false;

  try {
    // -------------------------------------------------------------------------
    // Playback (creates AudioContext — MUST be user-gesture-initiated)
    // -------------------------------------------------------------------------

    playback = new Playback({
      onEvent: (e) => log("[playback]", e.type),
    });
    await playback.start();

    window.playback = playback;
    window.ws = ws;
    window.mic = mic;
    window.bargeIn = bargeIn;

    // -------------------------------------------------------------------------
    // WebSocket
    // -------------------------------------------------------------------------

    ws = new WSClient({
      url: `ws://localhost:8000/ws`,
      playback,

      onJson: (msg) => {
        log("[ws] JSON:", msg.type);

        // Capture run_id for stale TTS frame filtering
        // (verify backend message contract)
        if (msg.type === "SESSION_INIT" && msg.run_id != null) {
          ws.setActiveTtsRunId(msg.run_id);
        }


      },

      onStatus: (status) => {
        log("[ws]", status);
        if (status === "UP") {
        ws.sendJson("MIC_START");
        }
      },

      onEvent: (e) => log("[ws]", e.type),
    });

    ws.connect();

    // -------------------------------------------------------------------------
    // Mic capture (client-side resample → PCM16 → framing)
    // -------------------------------------------------------------------------

    mic = new MicCapture({
      onFrame: (frame) => {
        ws.sendMicFrame(frame);
      },
    });

    await mic.start();

    // Wire WS backpressure → mic pause/resume
    ws.setMicControl({
      pause: () => mic.pause(),
      resume: () => mic.resume(),
    });

    // -------------------------------------------------------------------------
    // Barge-in detection
    // -------------------------------------------------------------------------
    // IMPORTANT (Phase 5):
    // This creates a SECOND getUserMedia call (separate from MicCapture).
    // This is an intentional Phase-5 compromise.
    //
    // Phase 6 TODO:
    // - Share mic MediaStream between capture + barge-in
    // - Avoid duplicate device access / permission prompts
    // -------------------------------------------------------------------------

    bargeIn = new BargeInDetector({
      onBargeIn: async () => {
        log("[barge-in] ⚡ DETECTED");
        await playback.hardStopAndRestart();
        ws.sendJson("BARGE_IN");
      },
    });

    // Uses existing AudioContext; does NOT create its own
    await bargeIn.start(playback.audioContext);

    log("✓ System started");
  } catch (err) {
    log("[start] FAILED:", err.message);

    // Best-effort cleanup on partial startup
    try { bargeIn?.stop(); } catch {}
    try { mic?.stop(); } catch {}
    try { ws?.disconnect(); } catch {}
    try { playback?.stop(); } catch {}

    bargeIn = mic = ws = playback = null;

    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
};

// -----------------------------------------------------------------------------
// Stop
// -----------------------------------------------------------------------------

stopBtn.onclick = async () => {
  stopBtn.disabled = true;
  startBtn.disabled = false;

  try {
    if (bargeIn) {
      bargeIn.stop();
      bargeIn = null;
    }

    if (mic) {
      mic.stop();
      mic = null;
    }

    if (ws) {
      ws.sendJson("MIC_STOP");
      ws.disconnect();
      ws = null;
    }

    if (playback) {
      playback.stop();
      playback = null;
    }

    log("✓ System stopped");
  } catch (err) {
    log("[stop] error:", err.message);
  }
};

window.testTone = async () => {
  if (!window.playback) {
    console.error("playback not initialized");
    return;
  }

  const samples = new Float32Array(320);
  for (let i = 0; i < samples.length; i++) {
    samples[i] = Math.sin(i / 10);
  }

  const pcm16 = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    pcm16[i] = samples[i] * 0x7fff;
  }

  playback.enqueueFrame({
    seqNum: 1,
    pcmBytes: new Uint8Array(pcm16.buffer),
    receivedAt: performance.now(),
  });

  console.log("testTone enqueued");
};