"""
Application configuration.

Responsibilities:
- Load deployment-specific configuration
- Read environment variables
- Provide a typed, immutable config object

Non-responsibilities:
- No orchestration logic
- No protocol constants
- No runtime mutation
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """
    Immutable application configuration.

    Constructed once at process startup.
    Passed downward to session/gateway/bootstrap code. (currently gateway)
    """

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    env: str
    log_level: str

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    deepgram_api_key: str | None

    # ------------------------------------------------------------------
    # LLM configuration
    # ------------------------------------------------------------------

    llm_provider: str
    llm_model: str
    openai_api_key: str | None

    groq_api_key: str | None

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    enable_json_logs: bool

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------
    tts_provider: str
    speechmatics_api_key: str | None
    speechmatics_voice: str | None
    elevenlabs_api_key: str | None
    elevenlabs_voice_id: str | None
    elevenlabs_model_id: str | None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_env() -> AppConfig:
        """
        Load configuration from environment variables.

        Raises:
            KeyError if required variables are missing.
        """
        return AppConfig(
            env=os.environ.get("ENV", "dev"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),

            llm_provider=os.environ.get("LLM_PROVIDER", "openai"),
            llm_model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            groq_api_key=os.environ.get("GROQ_API_KEY"),

            enable_json_logs=os.environ.get("ENABLE_JSON_LOGS", "1") == "1",
            speechmatics_api_key=os.environ.get("SPEECHMATICS_API_KEY"),
            tts_provider=os.environ.get("TTS_PROVIDER", "speechmatics"),
            speechmatics_voice=os.environ.get("SPEECHMATICS_VOICE", "sarah"),
            elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY"),
            elevenlabs_voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            elevenlabs_model_id=os.environ.get("ELEVENLABS_MODEL_ID", "eleven_turbo_v2"),
            deepgram_api_key=os.environ.get("DEEPGRAM_API_KEY")
        )
