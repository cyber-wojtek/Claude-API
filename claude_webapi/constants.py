"""
Model constants and other enumerations for claude_webapi.
"""
from __future__ import annotations

from enum import Enum


class Model(Enum):
    """
    Named Claude models available on Claude.ai.

    Pass any member to ``generate_content`` or ``start_chat`` via the
    ``model`` argument.  You can also pass a raw model-string directly
    (e.g. ``model="claude-opus-4-6"``).

    Example::

        from claude_webapi.constants import Model
        response = await client.generate_content("Hi", model=Model.SONNET)
    """

    # ── Claude 4.x ────────────────────────────────────────────────────────
    SONNET       = "claude-sonnet-4-6"
    OPUS         = "claude-opus-4-6"
    HAIKU        = "claude-haiku-4-5-20251001"

    # ── Claude 3.x (legacy, still accepted) ──────────────────────────────
    SONNET_3_7   = "claude-3-7-sonnet-20250219"
    SONNET_3_5   = "claude-3-5-sonnet-20241022"
    HAIKU_3_5    = "claude-3-5-haiku-20241022"
    OPUS_3       = "claude-3-opus-20240229"

    # ── Default (unspecified — Claude.ai picks) ───────────────────────────
    DEFAULT      = "claude-sonnet-4-6"

    @property
    def model_name(self) -> str:
        return self.value


#: Default model used when no model is specified.
DEFAULT_MODEL: str = Model.DEFAULT.value

CLAUDE_BASE_URL: str = "https://claude.ai"

