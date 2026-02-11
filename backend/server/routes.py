"""
Route registration for voice agent API.

Responsibilities:
- Define HTTP and WebSocket endpoints
- Wire gateway to WebSocket lifecycle
- Pull dependencies from app.state
"""

from __future__ import annotations

import base64
import json
import struct


from fastapi import WebSocket, WebSocketDisconnect, FastAPI, Response
from openai import AsyncOpenAI

from observability.logger import log_event
from session.gateway import SessionGateway, GatewayResult

from audio.telephony_codec import TelephonyCodec


def register_routes(app: FastAPI) -> None:
    """Register all routes on the FastAPI app."""
    @app.get("/health")
    async def health() -> dict[str, str]: # pyright: ignore[reportUnusedFunction]
        return {"status": "ok"}

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None: # pyright: ignore[reportUnusedFunction]
        await ws.accept()

        # OpenAI client is pulled from app state
        openai_client: AsyncOpenAI = app.state.openai_client

        gateway = SessionGateway(
            config=app.state.config,
            openai_client=openai_client,
        )

        try:
            result = await gateway.on_ws_connect()
            await _flush_gateway_result(ws, result)

            while True:
                msg = await ws.receive()

                if "text" in msg:
                    result = await gateway.on_json_message(msg["text"])
                    await _flush_gateway_result(ws, result)

                elif "bytes" in msg:
                    result = await gateway.on_binary_message(msg["bytes"])
#                    print("Gateway: result len = " ,len(result.outbound_binary))
                    await _flush_gateway_result(ws, result)

        except WebSocketDisconnect:
            await gateway.on_ws_disconnect(reason="client_disconnect")

        except Exception as exc:  # pylint: disable=broad-exception-caught
            log_event({
                "event_type": "WS_FATAL_ERROR",
                "session_id": gateway.session.session_id if gateway.session else None,
                "exception": type(exc).__name__,
                "message": str(exc),
            })
            await gateway.on_ws_disconnect(reason="server_error")

    @app.post("/twilio/incoming-call")
    async def twilio_incoming_call() -> Response: # pyright: ignore[reportUnusedFunction]
        twiml = """
        <Response>
            <Connect>
                <Stream url="wss://otilia-unchurlish-art.ngrok-free.dev/twilio/media-stream" />
            </Connect>
        </Response>
        """
        return Response(content=twiml, media_type="text/xml")

    @app.websocket("/twilio/media-stream")
    async def twilio_media_stream(ws: WebSocket) -> None:  # pyright: ignore[reportUnusedFunction]
        await ws.accept()

        openai_client: AsyncOpenAI = app.state.openai_client

        gateway = SessionGateway(
            config=app.state.config,
            openai_client=openai_client,
        )

        decoder = TelephonyCodec()
        seq = 0
        stream_sid: str | None = None
        aligner = FrameAligner()

        try:
            result = await gateway.on_ws_connect()
            for msg in result.outbound_json:
                await ws.send_text(json.dumps(msg))

            while True:
                msg = await ws.receive_text()
                data = json.loads(msg)

                event_type = data.get("event")

                if event_type == "media":
                    payload_b64 = data["media"]["payload"]
                    mulaw_audio = base64.b64decode(payload_b64)

                    pcm16k = decoder.decode(mulaw_audio)

                    if pcm16k:
                        framed = struct.pack("<I", seq) + pcm16k
                        seq += 1

                        result = await gateway.on_binary_message(framed)
                        # Send control JSON (still valid for Twilio WS)
                        for msg in result.outbound_json:
                            if msg.get("type") == "AUDIO_STOP" and stream_sid is not None:
                                await ws.send_text(json.dumps({
                                    "event": "clear",
                                    "streamSid": stream_sid,
                                }))
                                aligner.clear_buffer()


                        # -------------------------------------------------
                        # TELEPHONY PLAYBACK (TTS → Twilio)
                        # -------------------------------------------------
                        if stream_sid is not None:
                            for frame in result.outbound_binary:
                                pcm_audio = frame[-640:]
                                assert len(pcm_audio) == 640, len(pcm_audio)

                                # Get aligned 640-byte chunks
                                for aligned_frame in aligner.add(pcm_audio):
                                    mulaw = decoder.encode(aligned_frame)

                                    # Verify clean encoding
                                    assert len(mulaw) == 160, (
                                        f"Expected 160 bytes, "
                                        f"got {len(mulaw)}"
                                    )

                                    if mulaw:
                                        payload = base64.b64encode(mulaw).decode()
                                        await ws.send_text(json.dumps({
                                            "event": "media",
                                            "streamSid": stream_sid,
                                            "media": {"payload": payload},
                                        }))

                elif event_type == "start":
                    stream_sid = data["start"]["streamSid"]
                    print("Twilio stream started")

                elif event_type == "stop":
                    print("Twilio stream stopped")
                    break

        except WebSocketDisconnect:
            await gateway.on_ws_disconnect(reason="twilio_disconnect")

        except Exception as exc: # pylint: disable=broad-exception-caught
            log_event({
                "event_type": "TWILIO_STREAM_FATAL_ERROR",
                "session_id": gateway.session.session_id if gateway.session else None,
                "exception": type(exc).__name__,
                "message": str(exc),
            })
            await gateway.on_ws_disconnect(reason="twilio_error")


async def _flush_gateway_result(
    ws: WebSocket,
    result: GatewayResult,
) -> None:
    for msg in result.outbound_json:
        await ws.send_text(json.dumps(msg))


    for frame in result.outbound_binary:
        await ws.send_bytes(frame)

class FrameAligner:
    """Rechunk audio to exact 20ms boundaries without loss."""
    FRAME_SIZE = 640  # 320 samples × 2 bytes @ 16kHz = 20ms

    def __init__(self):
        self._buffer = b""

    def add(self, pcm16_bytes: bytes) -> list[bytes]:
        """Add audio and return complete 20ms frames."""
        self._buffer += pcm16_bytes
        frames: list[bytes] = []

        while len(self._buffer) >= self.FRAME_SIZE:
            frames.append(self._buffer[:self.FRAME_SIZE])
            self._buffer = self._buffer[self.FRAME_SIZE:]

        return frames

    def clear_buffer(self):
        """Clear the aligner's internal buffer"""
        self._buffer = b""
