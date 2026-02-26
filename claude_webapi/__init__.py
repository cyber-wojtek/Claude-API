"""
claude_webapi — Async Python wrapper for the Claude.ai web app.

Quick start::

    import asyncio
    from claude_webapi import ClaudeClient

    async def main():
        client = ClaudeClient("sk-ant-…", "your-org-uuid")
        await client.init()

        response = await client.generate_content("Hello Claude!")
        print(response.text)

    asyncio.run(main())
"""

from .client import ClaudeClient
from .constants import Model
from .exceptions import (
    APIError,
    AuthenticationError,
    ClaudeWebAPIError,
    ConversationNotFoundError,
    FileUploadError,
    QuotaExceededError,
    TimeoutError,
)
from .session import ChatSession
from .types import Candidate, Image, ModelOutput

import logging as _logging


def set_log_level(level: str) -> None:
    """
    Configure the ``claude_webapi`` logger.

    Calling this for the first time removes any existing handlers on the
    ``claude_webapi`` logger so the new level takes effect cleanly.

    Parameters
    ----------
    level:
        One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``,
        ``"CRITICAL"``.

    Example::

        from claude_webapi import set_log_level
        set_log_level("DEBUG")
    """
    log = _logging.getLogger("claude_webapi")
    log.handlers.clear()
    handler = _logging.StreamHandler()
    handler.setFormatter(
        _logging.Formatter("%(asctime)s  %(levelname)-5s  %(name)s  %(message)s")
    )
    log.addHandler(handler)
    log.setLevel(getattr(_logging, level.upper(), _logging.INFO))


__all__ = [
    "ClaudeClient",
    "ChatSession",
    "Model",
    "ModelOutput",
    "Candidate",
    "Image",
    "ClaudeWebAPIError",
    "APIError",
    "AuthenticationError",
    "ConversationNotFoundError",
    "FileUploadError",
    "QuotaExceededError",
    "TimeoutError",
    "set_log_level",
]

__version__ = "1.0.0"

