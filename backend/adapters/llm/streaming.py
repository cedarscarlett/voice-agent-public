"""LLM Adapter"""
from __future__ import annotations

import os #TODO: remove these
import asyncio
import time
import json
from collections.abc import Awaitable, Callable
from typing import Any

from orchestrator.enums.service import Service
from orchestrator.events import (
    Event,
    EventType,
    LLMDone,
    LLMError,
    LLMToken,
    LLMToolCall,
)

print("PROCESS PID:", os.getpid())


class StreamingLLMAdapter:
    """
    Concrete streaming LLM adapter.

    Design notes:
    - One adapter instance may serve multiple sequential runs.
    - Each run is tracked independently via run_id → asyncio.Task.
    - Adapter is responsible ONLY for:
        - Talking to the LLM provider
        - Streaming tokens
        - Emitting LLM_* events
    - Adapter does NOT:
        - Retry
        - Chunk text
        - Manage timers
        - Decide orchestration outcomes
    """

    def __init__(
        self,
        *,
        emit_event: Callable[[Event], Awaitable[None]],
        client: Any,
        model: str,
        session_id: str,
        provider: str
    ) -> None:
        """
        Args:
            emit_event:
                Callback used to emit events into the runtime/event loop.
                Must be non-blocking.
            client:
                Vendor client (e.g., OpenAI, vLLM, etc.).
            model:
                Model identifier string.
            session_id:
                Session identifier for logging/correlation.
        """
        self._emit_event = emit_event
        self._client = client
        self._model = model
        self._session_id = session_id
        self._provider = provider

        # One task per active run_id
        self._active_tasks: dict[int, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_completion(self, *, run_id: int, messages: list[dict[str, str]]) -> None:
        """Call the completions API"""
        print(
            "LLM start_completion called:",
            run_id,
        )
        if run_id in self._active_tasks:
            return

        task = asyncio.create_task(self._run_stream(run_id, messages))
        self._active_tasks[run_id] = task

        # Cleanup when done
        def _cleanup(_: asyncio.Task[None]) -> None:
            self._active_tasks.pop(run_id, None)

        task.add_done_callback(_cleanup)

        # IMPORTANT: return immediately (do NOT await task)
        return

    async def cancel(self, run_id: int) -> None:
        """
        Best-effort cancellation.

        Semantics:
        - Idempotent.
        - Silent if run_id is stale or already completed.
        - Cancellation may result in zero terminal events (allowed by spec).
        """
        task = self._active_tasks.get(run_id)
        if task is None:
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def force_reset(self) -> None:
        """
        Hard reset: cancel all active LLM runs immediately.

        This is a last-resort safety valve used after cancellation
        acknowledgement timeouts.
        """
        for task in self._active_tasks.values():
            task.cancel()

        self._active_tasks.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run_stream(
        self,
        run_id: int,
        messages: list[dict[str, str]],
    ) -> None:
        """
        Internal streaming task.

        Guarantees:
        - Emits events only for its own run_id
        - Emits at most one terminal event
        """
        full_text = [] # TODO: remove this
        print(
            "LLM _run_stream ENTRY",
        )
        print("LLM CALL START", {
            "t_ns": time.monotonic_ns(),
            "run_id": run_id,
            "model" : self._model
        })
        try:
            kwargs = dict(
                model=self._model,
                messages=messages,
                stream=True,
            )
            if self._provider == "openai":
                kwargs["service_tier"] = "priority"

            stream = await self._client.chat.completions.create(**kwargs)
            print("LLM STREAM OBJECT RETURNED", {
                "t_ns": time.monotonic_ns(),
                "run_id": run_id,
            })

            first = True
            speech_mode = True
            tool_buffer = ""

            async for chunk in stream:
                if first:
                    print("LLM RAW CHUNK REPR:", repr(chunk))
                    first = False
                delta = self._extract_delta(chunk)
                if delta:
                    full_text.append(delta) # pyright: ignore[reportUnknownMemberType]
                if not delta:
                    continue

                if speech_mode:
                    if "@" in delta:
                        speech_part, tool_part = delta.split("@", 1)

                        if speech_part:
                            await self._emit_event(
                                LLMToken(
                                    event_type=EventType.LLM_TOKEN,
                                    ts_ms=self._now_ms(),
                                    service=Service.LLM,
                                    run_id=run_id,
                                    delta=speech_part,
                                )
                            )

                        speech_mode = False
                        tool_buffer = tool_part
                        continue

                    await self._emit_event(
                        LLMToken(
                            event_type=EventType.LLM_TOKEN,
                            ts_ms=self._now_ms(),
                            service=Service.LLM,
                            run_id=run_id,
                            delta=delta,
                        )
                    )
                else:
                    tool_buffer += delta
            print("LLM FINAL TEXT READY", {
                "t_ns": time.monotonic_ns(),
                "run_id": run_id,
            })

            final_text = "".join(full_text) # pyright: ignore[reportUnknownArgumentType]
            print("LLM FINAL TEXT REPR:", repr(final_text))
            print("LLM FINAL TEXT LEN:", len(final_text))

            if tool_buffer:
                try:
                    data = json.loads(tool_buffer)
                    if "tool" not in data:
                        raise ValueError("missing 'tool' field")

                    tool = data["tool"]
                    args = {k: v for k, v in data.items() if k != "tool"}

                    await self._emit_event(
                        LLMToolCall(
                            event_type=EventType.LLM_TOOL_CALL,
                            ts_ms=self._now_ms(),
                            service=Service.LLM,
                            run_id=run_id,
                            tool=tool,
                            args=args,
                        )
                    )
                except Exception as exc:
                    await self._emit_event(
                        LLMError(
                            event_type=EventType.LLM_ERROR,
                            ts_ms=self._now_ms(),
                            service=Service.LLM,
                            run_id=run_id,
                            reason=f"malformed_tool_call: {type(exc).__name__}: {exc}",
                        )
                    )
                    return

            # Normal completion
            await self._emit_event(
                LLMDone(
                    event_type=EventType.LLM_DONE,
                    ts_ms=self._now_ms(),
                    service=Service.LLM,
                    run_id=run_id,
                )
            )

        except asyncio.CancelledError:
            # Expected during barge-in or explicit cancellation
            # Cleanup happens in finally block of start_completion
            pass

        except Exception as exc:  # pylint: disable=broad-exception-caught
            await self._emit_event(
                LLMError(
                    event_type=EventType.LLM_ERROR,
                    ts_ms=self._now_ms(),
                    service=Service.LLM,
                    run_id=run_id,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_delta(chunk: Any) -> str:
        """
        Extract token delta from vendor response (OpenAI format).
        """
        try:
            delta = chunk.choices[0].delta
            return delta.content or ""
        except (AttributeError, IndexError):
            return ""

    @staticmethod
    def _now_ms() -> int:
        """
        Wall-clock timestamp in milliseconds.

        Orchestrator is responsible for monotonic timing;
        adapters only provide coarse timestamps for ordering/logging.
        """
        return int(time.time() * 1000)
