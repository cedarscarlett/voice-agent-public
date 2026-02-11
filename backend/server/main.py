"""
FastAPI WebSocket server for real-time voice agent.

Responsibilities:
- Accept WebSocket connections
- Create SessionGateway per connection
- Route messages between client and gateway
- Handle connection lifecycle
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from observability.logger import log_event
from session.gateway import SessionGateway, GatewayResult


# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------

app = FastAPI(title="Voice Agent API")

# CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set")

openai_client = AsyncOpenAI(api_key=openai_api_key)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint for load balancers."""
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """
    Main WebSocket endpoint for voice sessions.

    One connection = one session = one gateway.
    """
    await ws.accept()

    # Create gateway (one per connection)
    gateway = SessionGateway(
        config=app.state.config,
        openai_client=openai_client,
    )

    try:
        # ---- CONNECT ----
        result = await gateway.on_ws_connect()
        await _flush_gateway_result(ws, result)

        # ---- MAIN LOOP ----
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.receive":
                if "text" in msg:
                    result = await gateway.on_json_message(msg["text"])
                    await _flush_gateway_result(ws, result)

                elif "bytes" in msg:
                    result = await gateway.on_binary_message(msg["bytes"])
                    await _flush_gateway_result(ws, result)

            elif msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()

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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

async def _flush_gateway_result(
    ws: WebSocket,
    result: GatewayResult,
) -> None:
    """
    Send all outbound messages produced by gateway.

    Sends JSON messages first, then binary frames.
    """
    for msg in result.outbound_json:
        await ws.send_text(json.dumps(msg))

    for frame in result.outbound_binary:
        await ws.send_bytes(frame)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,  # Dev mode only
    )
