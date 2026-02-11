"""
FastAPI app factory.

Responsibilities:
- Create and configure FastAPI app
- Set up middleware
- Initialize shared resources (OpenAI client)
- Register routes
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from config import AppConfig

from server.routes import register_routes


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This is the app factory pattern that allows:
    - Testing with different configurations
    - Environment-specific setup
    - ASGI server compatibility
    """
    config = AppConfig.load_from_env()

    app = FastAPI(title="Voice Agent API")

    app.state.config = config

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten later
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create OpenAI client ONCE per process
    openai_api_key = config.openai_api_key
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    app.state.openai_client = build_llm_client(
        config=config
    )

    # Routes
    register_routes(app)

    return app

def build_llm_client(config: AppConfig) -> AsyncOpenAI:
    """Build an LLM client with the provider selected by environment variables."""
    if config.llm_provider.lower() == "groq":
        return AsyncOpenAI(
            api_key=config.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    return AsyncOpenAI(api_key=config.openai_api_key)
