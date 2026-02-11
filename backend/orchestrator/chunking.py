"""
Pure TTS chunking logic (Spec §7).

This module contains NO side effects and NO timing primitives.
It is a deterministic function over:
- current text buffer
- elapsed time since chunk_timer started

All orchestration (timers, scheduling, retries) lives elsewhere.

IMPORTANT CONTRACT WITH REDUCER / RUNTIME:

- This module never starts, stops, or resets timers.
- When `send=False` is returned due to Trigger B/C firing with
  `len(buffer) < TTS_MIN_CHUNK_LEN_CHARS`, the caller MUST still
  reset the chunk timer (Spec §7) to ensure progress with slow
  token streams.
- When `remainder == ""`, the caller is responsible for stopping
  chunk-timer polling, since the buffer is now empty.
"""

from __future__ import annotations

from dataclasses import dataclass

from spec import (
    TTS_MIN_CHUNK_LEN_CHARS,
    TTS_SIZE_TRIGGER_CHARS,
    TTS_HARD_CAP_CHARS,
    TTS_LOOKBACK_CHARS,
    TTS_CHUNK_TIME_MS,
    TTS_PREFERRED_BREAK_CHARS,
    TTS_WHITESPACE_BREAK_CHARS
)


# =============================================================================
# Chunking Decision
# =============================================================================

@dataclass(frozen=True)
class ChunkDecision:
    """
    Result of a chunking evaluation.

    If send is False, all other fields are undefined and must be ignored.
    """
    send: bool
    send_text: str | None = None
    remainder: str | None = None
    forced_mid_word: bool = False


# =============================================================================
# Public API
# =============================================================================

def evaluate_chunk(
    *,
    buffer: str,
    elapsed_ms: int,
) -> ChunkDecision:
    """
    Evaluate whether a TTS chunk should be emitted.

    Args:
        buffer:
            Current accumulated LLM text buffer (non-empty).
        elapsed_ms:
            Time elapsed since buffer transitioned from empty → non-empty.

    Returns:
        ChunkDecision describing whether to send a chunk and how to split.

    Notes:
        - This function is PURE and deterministic.
        - Timer reset semantics are the responsibility of the caller.
    """
    print(
        "CHUNK EVAL:",
        {
            "buffer_len": len(buffer),
            "buffer_repr": repr(buffer[-80:]),  # tail only
            "elapsed_ms": elapsed_ms,
        }
    )
    if not buffer:
        return ChunkDecision(send=False)

    # ------------------------------------------------------------
    # Guard: ignore whitespace-only buffers
    # ------------------------------------------------------------
    if not buffer.strip():
        print("CHUNK IGNORE: whitespace-only buffer")
        return ChunkDecision(send=False)

    # ------------------------------------------------------------
    # Trigger A: Sentence boundary (Spec §7)
    #
    # Send immediately, even if < MIN_CHUNK_LEN_CHARS, but ONLY up
    # to and including the FIRST sentence boundary.
    # ------------------------------------------------------------
    boundary_decision = _split_at_sentence_boundary(buffer)
    if boundary_decision is not None:
        return boundary_decision

    # ------------------------------------------------------------
    # Trigger D: Hard cap (must send, with backtrack)
    # ------------------------------------------------------------
    if len(buffer) >= TTS_HARD_CAP_CHARS:
        return _split_with_backtrack(buffer)

    # ------------------------------------------------------------
    # Trigger B: Size threshold
    # Trigger C: Time threshold
    # ------------------------------------------------------------
    size_triggered = len(buffer) >= TTS_SIZE_TRIGGER_CHARS
    time_triggered = elapsed_ms >= TTS_CHUNK_TIME_MS

    if size_triggered or time_triggered:
        if len(buffer) >= TTS_MIN_CHUNK_LEN_CHARS:
            split = _split_at_last_break(buffer)
            if split is not None:
                print(
                    "CHUNK TRIGGER B/C (natural split)",
                    {"buffer_len": len(buffer)},
                )
                return split

            # Fallback: no break char found
            print(
                "CHUNK TRIGGER B/C (no break found)",
                {"buffer_len": len(buffer)},
            )
            return ChunkDecision(
                send=True,
                send_text=buffer.lstrip(),
                remainder="",
                forced_mid_word=False,
            )

        # Buffer too short → do not send.
        # Caller MUST reset the chunk timer anyway (Spec §7).
        return ChunkDecision(send=False)

    # ------------------------------------------------------------
    # No trigger fired
    # ------------------------------------------------------------
    print(
        "CHUNK NO TRIGGER",
        {
            "buffer_len": len(buffer),
            "elapsed_ms": elapsed_ms,
        }
    )
    return ChunkDecision(send=False)


# =============================================================================
# Helpers
# =============================================================================

def _split_at_sentence_boundary(buffer: str) -> ChunkDecision | None:
    """
    Split buffer at the FIRST sentence boundary, if present.

    Sentence boundaries (Spec §7):
    - '.', '!', '?', or newline

    Returns:
        ChunkDecision if a boundary is found, otherwise None.
    """
    for ch in (".", "!", "?", "\n"):
        idx = buffer.find(ch)
        if idx != -1:
            split_idx = idx + 1  # include delimiter
            print(
                "CHUNK TRIGGER A (sentence boundary)",
                {
                    "char": buffer[idx],
                    "split_idx": split_idx,
                    "send_len": split_idx,
                    "remainder_len": len(buffer) - split_idx,
                }
            )
            return ChunkDecision(
                send=True,
                # Guard: prevent whitespace-only emission
                send_text=buffer[:split_idx].lstrip(),
                remainder=buffer[split_idx:],
                forced_mid_word=False,
            ) if buffer[:split_idx].strip() else None
    print("CHUNK TRIGGER A returning None")
    return None


def _split_at_last_break(buffer: str) -> ChunkDecision | None:
    print("_split_at_last_break entry")
    last_space_idx = None

    for i in range(len(buffer) - 1, -1, -1):
        split_len = i + 1
        ch = buffer[i]

        if ch in TTS_WHITESPACE_BREAK_CHARS and last_space_idx is None:
            if split_len >= TTS_MIN_CHUNK_LEN_CHARS:
                last_space_idx = i

        print("SCAN CHAR:", repr(ch), ord(ch))

        # Only search for preferred breaks while still >= MIN
        if split_len < TTS_MIN_CHUNK_LEN_CHARS:
            if last_space_idx is not None:
                break
            continue

        if ch in TTS_PREFERRED_BREAK_CHARS:
            return ChunkDecision(
                send=True,
                send_text=buffer[:split_len].lstrip(),
                remainder=buffer[split_len:],
                forced_mid_word=False,
            )

    if last_space_idx is not None:
        split_idx = last_space_idx + 1
        return ChunkDecision(
            send=True,
            send_text=buffer[:split_idx].lstrip(),
            remainder=buffer[split_idx:],
            forced_mid_word=False,
        )

    return None

def _split_with_backtrack(buffer: str) -> ChunkDecision:
    """
    Split buffer using lookback window when hard cap is exceeded.
    """
    print("_split_with_backtrack entry")
    print(
        "CHUNK TRIGGER D (hard cap)",
        {
            "buffer_len": len(buffer),
            "hard_cap": TTS_HARD_CAP_CHARS,
        }
    )

    lookback_start = max(0, len(buffer) - TTS_LOOKBACK_CHARS)
    lookback_region = buffer[lookback_start:]

    last_space_idx = None

    for i in range(len(lookback_region) - 1, -1, -1):
        split_len = lookback_start + i + 1
        ch = lookback_region[i]

        print("SCAN CHAR:", repr(ch), ord(ch))

        # Prefer punctuation only if resulting chunk >= MIN
        if (
            ch in TTS_PREFERRED_BREAK_CHARS
            and split_len >= TTS_MIN_CHUNK_LEN_CHARS
        ):
            return ChunkDecision(
                send=True,
                send_text=buffer[:split_len].lstrip(),
                remainder=buffer[split_len:],
                forced_mid_word=False,
            )

        # Track whitespace fallback ≥ MIN
        if (
            ch in TTS_WHITESPACE_BREAK_CHARS
            and last_space_idx is None
            and split_len >= TTS_MIN_CHUNK_LEN_CHARS
        ):
            last_space_idx = i

    # Whitespace fallback
    if last_space_idx is not None:
        split_idx = lookback_start + last_space_idx + 1
        return ChunkDecision(
            send=True,
            send_text=buffer[:split_idx].lstrip(),
            remainder=buffer[split_idx:],
            forced_mid_word=False,
        )

    # True hard-cap split
    split_idx = TTS_HARD_CAP_CHARS
    return ChunkDecision(
        send=True,
        send_text=buffer[:split_idx].lstrip(),
        remainder=buffer[split_idx:],
        forced_mid_word=True,
    )
