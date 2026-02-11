"""
ASGI entry point.

Used by uvicorn / gunicorn.
"""

from dotenv import load_dotenv

load_dotenv()

from server.app import create_app  # pylint: disable=wrong-import-position

app = create_app()
