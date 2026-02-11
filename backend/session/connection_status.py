"""
Connection status tracking for voice sessions.

Per spec §4: Connection lifecycle is tracked separately from state machine.
connection_status: DOWN | CONNECTING | UP

This is pure data owned by SessionGateway, not by orchestrator state.
"""
from enum import Enum

class ConnectionStatus(Enum):
    """
    Connection lifecycle status (spec §4).

    Separate from and independent of orchestrator State enum.
    IDLE can occur with any ConnectionStatus.
    """
    DOWN = "DOWN"           # Not connected
    CONNECTING = "CONNECTING"  # Attempting connection (with retry backoff)
    UP = "UP"              # Active WebSocket connection
