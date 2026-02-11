"""
Cancellation protocol runtime (v1).

Responsibilities:
- Implement Cancel/ACK protocol for ASR / LLM / TTS
- Start ACK timeout timers
- Emit CancelAck or CancelTimeout events
- Perform hard-reset hooks on timeout (via callback)

Non-responsibilities:
- NO retry logic
- NO state machine decisions
- NO run_id generation
- NO Start* commands
- NO knowledge of reducer transitions

This module is infrastructure only.
"""

from __future__ import annotations

import asyncio
from asyncio import Task
from dataclasses import dataclass
from typing import Callable, Awaitable

from orchestrator.enums.service import Service
from orchestrator.events import (
    Event,
    CancelTimeout,
    EventType
)
from spec import CANCEL_ACK_TIMEOUT_MS


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

EventSink = Callable[[Event], Awaitable[None]]
HardResetFn = Callable[[Service], None]
GetActiveRunFn = Callable[[Service], int]


@dataclass(frozen=True)
class _CancelKey:
    service: Service
    run_id: int


# ---------------------------------------------------------------------
# Cancellation Manager
# ---------------------------------------------------------------------

class CancellationManager:
    """
    Runtime manager for Cancel/ACK protocol.

    Lifecycle:
    1. Orchestrator emits CancelX(run_id)
    2. Runtime calls request_cancel(service, run_id)
    3. Manager starts ACK timeout timer
    4a. Adapter emits ACK -> notify_ack(...)
    4b. Timer fires -> emit CancelTimeout + hard reset

    This class never decides what happens next.
    """

    def __init__(
        self,
        *,
        emit_event: EventSink,
        hard_reset: HardResetFn,
        get_active_run: GetActiveRunFn,
    ) -> None:
        self._emit_event = emit_event
        self._hard_reset = hard_reset
        self._get_active_run = get_active_run

        self._timers: dict[_CancelKey, Task[None]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_cancel(self, *, service: Service, run_id: int) -> None:
        """
        Register a cancel request and start ACK timeout timer.

        Idempotent: duplicate calls for same (service, run_id) are ignored.
        """
        key = _CancelKey(service, run_id)
        if key in self._timers:
            return

        self._timers[key] = asyncio.create_task(
            self._ack_timeout_task(service=service, run_id=run_id)
        )

    def notify_ack(self, *, service: Service, run_id: int) -> None:
        """
        Cancel ACK timeout timer.

        The original CancelAck event from the adapter flows through
        handle_event() and reducer normally. This method only handles
        the protocol infrastructure (stopping the timeout timer).
        """
        key = _CancelKey(service, run_id)
        task = self._timers.pop(key, None)
        if task:
            task.cancel()



    def clear_all(self) -> None:
        """
        Cancel and clear all outstanding ACK timers.
        Used on session teardown.
        """
        for task in self._timers.values():
            task.cancel()
        self._timers.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _ack_timeout_task(self, *, service: Service, run_id: int) -> None:
        """
        Wait for ACK timeout; emit CancelTimeout on expiry.
        """
        try:
            await asyncio.sleep(CANCEL_ACK_TIMEOUT_MS / 1000.0)
        except asyncio.CancelledError:
            return

        key = _CancelKey(service, run_id)
        self._timers.pop(key, None)

        if self._get_active_run(service) == run_id:
            self._hard_reset(service)

        await self._emit_event(
            CancelTimeout(
                event_type=EventType.CANCEL_TIMEOUT,
                service=service,
                run_id=run_id,
                ts_ms=_now_ms(),
            )
        )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _now_ms() -> int:
    # Wall clock ordering only; reducer uses monotonic timers
    return int(asyncio.get_event_loop().time() * 1000)
