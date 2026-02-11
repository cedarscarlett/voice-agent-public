"""
Retry policy helpers (v1).

Purpose:
- Centralize retry rules from spec §11
- Keep reducer pure
- Allow runtime to make deterministic retry decisions

This module contains NO timers, NO async, NO side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from orchestrator.enums.service import Service

from spec import (
    ASR_RETRY_DELAYS_MS,
    LLM_INIT_RETRY_DELAY_MS,
    TTS_INIT_RETRY_DELAY_MS,
)


# =============================================================================
# Failure Types
# =============================================================================

class FailureType(str, Enum):
    """
    Failure classification used by retry policy.

    This enum captures *when* and *how* a failure occurred relative to
    stream lifecycle, which determines retry eligibility per spec §11.

    Semantics by category:

    INIT_ERROR:
        Failure occurred before any output was produced
        (e.g., connection failure, request rejected).
        Eligible for init-only retry (service-dependent).

    FIRST_TOKEN_TIMEOUT:
        No LLM token was received within the first-token timeout window.
        Treated as an init failure.
        Eligible for init-only retry (LLM only).

    FIRST_AUDIO_TIMEOUT:
        No TTS audio frame was received within the first-audio timeout window.
        Treated as an init failure.
        Eligible for init-only retry (TTS only).

    MID_STREAM_ERROR:
        Failure occurred after partial output was already produced.
        Never retried to avoid duplicated or inconsistent output.

    STALL_TIMEOUT:
        Stream stalled after initial output (no tokens/audio for timeout window).
        Considered a mid-stream failure.
        Never retried.

    Notes:
    - Cancellation is NOT a failure type and must never trigger retries.
    - Retry decisions are made purely from (service, FailureType, attempt).
    """

    INIT_ERROR = "init_error"
    FIRST_TOKEN_TIMEOUT = "first_token_timeout"
    FIRST_AUDIO_TIMEOUT = "first_audio_timeout"
    MID_STREAM_ERROR = "mid_stream_error"
    STALL_TIMEOUT = "stall_timeout"


# =============================================================================
# Retry State
# =============================================================================

@dataclass(frozen=True)
class RetryAttempt:
    """
    Immutable retry attempt counter.

    Semantics:
    - attempt == 0 represents the initial attempt (no retry yet).
    - attempt >= 1 represents the Nth retry attempt.
    - This value is used only for retry policy decisions
      (max attempts, backoff delay), never for control flow.
    """
    attempt: int


def next_attempt(current: RetryAttempt) -> RetryAttempt:
    """
    Advance to the next retry attempt.

    Returns a new RetryAttempt with attempt incremented by 1.

    Notes:
    - This function is pure and side-effect free.
    - Retry state is immutable by design to prevent accidental
      in-place mutation across runs or services.
    """
    return RetryAttempt(attempt=current.attempt + 1)


# =============================================================================
# Policy
# =============================================================================

def max_attempts(service: Service, failure: FailureType) -> int:
    """
    Maximum retry attempts (excluding the initial attempt).

    Spec §11:
    - ASR: 3 retries
    - LLM/TTS init: 1 retry
    - LLM/TTS mid-stream: 0 retries
    """
    if service is Service.ASR:
        return 3

    if service is Service.LLM:
        if failure in (
            FailureType.INIT_ERROR,
            FailureType.FIRST_TOKEN_TIMEOUT,
        ):
            return 1
        return 0

    if service is Service.TTS:
        if failure in (
            FailureType.INIT_ERROR,
            FailureType.FIRST_AUDIO_TIMEOUT,
        ):
            return 1
        return 0

    return 0


def should_retry(
    *,
    service: Service,
    failure: FailureType,
    attempt: RetryAttempt,
) -> bool:
    """
    Returns True if a retry is allowed.

    attempt = number of retries already performed
    """
    return attempt.attempt < max_attempts(service, failure)


# =============================================================================
# Delay Calculation
# =============================================================================

def get_retry_delay_ms(
    *,
    service: Service,
    attempt: RetryAttempt,
) -> int:
    """
    Returns delay before retry attempt N.

    Spec §11:
    ASR: exponential backoff
    LLM/TTS init: fixed
    """
    if service is Service.ASR:
        # Clamp attempt index to last backoff slot
        idx = min(attempt.attempt, len(ASR_RETRY_DELAYS_MS) - 1)
        return ASR_RETRY_DELAYS_MS[idx]

    if service is Service.LLM:
        return LLM_INIT_RETRY_DELAY_MS

    if service is Service.TTS:
        return TTS_INIT_RETRY_DELAY_MS

    return 0

def reset_attempt() -> RetryAttempt:
    """Returns a fresh retry attempt counter."""
    return RetryAttempt(attempt=0)
