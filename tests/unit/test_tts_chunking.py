# tests/unit/test_tts_chunking.py

from orchestrator.chunking import evaluate_chunk
from spec import (
    TTS_HARD_CAP_CHARS,
    TTS_LOOKBACK_CHARS,
)


def test_sentence_boundary_splits_at_first_boundary():
    buf = "Hello world. This should stay buffered."
    decision = evaluate_chunk(buffer=buf, elapsed_ms=0)

    assert decision.send is True
    assert decision.send_text == "Hello world."
    assert decision.remainder == " This should stay buffered."
    assert decision.forced_mid_word is False


def test_sentence_boundary_works_with_short_text():
    buf = "Hi!"
    decision = evaluate_chunk(buffer=buf, elapsed_ms=0)

    assert decision.send is True
    assert decision.send_text == "Hi!"
    assert decision.remainder == ""


def test_hard_cap_forces_split_when_no_break_chars():
    buf = "x" * (TTS_HARD_CAP_CHARS + 5)

    decision = evaluate_chunk(
        buffer=buf,
        elapsed_ms=0,
    )

    assert decision.send is True
    assert decision.send_text is not None
    assert len(decision.send_text) <= TTS_HARD_CAP_CHARS
    assert decision.forced_mid_word is True


def test_hard_cap_prefers_break_char_within_lookback():
    # Put a break char inside lookback window
    prefix = "a" * (TTS_HARD_CAP_CHARS - TTS_LOOKBACK_CHARS)
    lookback = "b" * (TTS_LOOKBACK_CHARS - 2) + " "
    buf = prefix + lookback + "TAIL"

    decision = evaluate_chunk(
        buffer=buf,
        elapsed_ms=0,
    )

    assert decision.send is True
    assert decision.forced_mid_word is False
    assert decision.send_text is not None
    assert decision.send_text.endswith(" ")
    assert decision.remainder == "TAIL"


def test_no_empty_chunks_emitted():
    decision = evaluate_chunk(buffer="", elapsed_ms=0)
    assert decision.send is False


def test_max_chars_invariant():
    buf = "x" * (TTS_HARD_CAP_CHARS * 2)

    decision = evaluate_chunk(
        buffer=buf,
        elapsed_ms=0,
    )

    assert decision.send is True
    assert decision.send_text is not None
    assert len(decision.send_text) <= TTS_HARD_CAP_CHARS
