"""
Conversation context serialization for LLM consumption.

Responsibilities:
- Convert system prompt + conversation context + current user text
  into LLM-ready message format.

Non-responsibilities:
- No truncation logic
- No turn storage
- No logging
- No orchestration decisions
"""

from __future__ import annotations

from context.conversation import ConversationContext


def serialize_for_llm(
    *,
    system_prompt: str,
    context: ConversationContext,
    user_text: str,
) -> list[dict[str, str]]:
    """
    Serialize conversation context into LLM message format.

    Output format:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
        {"role": "user", "content": "<current user text>"},
    ]

    Rules:
    - System prompt is always first
    - Stored context turns come next (already truncated)
    - Current user text is appended last as a fresh user turn
    """
    messages: list[dict[str, str]] = []

    # System prompt always comes first
    messages.append({
        "role": "system",
        "content": system_prompt,
    })

    # Historical context (already policy-enforced)
    messages.extend(context.serialize())

    # Current user input (not yet part of stored context)
    messages.append({
        "role": "user",
        "content": user_text,
    })

    return messages
