"""
ChatSession — multi-turn conversation wrapper for claude_webapi.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from .types import ModelOutput

if TYPE_CHECKING:
    from .client import ClaudeClient


class ChatSession:
    """
    Represents an ongoing multi-turn conversation with Claude.

    Do not instantiate directly — use :meth:`ClaudeClient.start_chat`.

    Example::

        chat = client.start_chat()
        reply1 = await chat.send_message("Tell me a joke.")
        reply2 = await chat.send_message("Explain why it's funny.")
    """

    def __init__(
        self,
        client: "ClaudeClient",
        conversation_id: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        metadata: dict | None = None,
    ):
        self._client          = client
        self._model           = model
        self._system_prompt   = system_prompt
        self._parent_uuid     = "00000000-0000-4000-8000-000000000000"
        self._current_candidate_index: int = 0

        # Restore or create conversation
        if metadata:
            self._conv_id       = metadata["conversation_id"]
            self._parent_uuid   = metadata.get("parent_message_uuid", self._parent_uuid)
            self._last_response : ModelOutput | None = None
        else:
            self._conv_id       = str(uuid.uuid4())
            self._last_response : ModelOutput | None = None

    # ── public metadata ────────────────────────────────────────────────────

    @property
    def cid(self) -> str:
        """Conversation UUID."""
        return self._conv_id

    @property
    def metadata(self) -> dict:
        """
        Serialisable dict that can be passed back to
        :meth:`ClaudeClient.start_chat` to resume this conversation.
        """
        return {
            "conversation_id":     self._conv_id,
            "parent_message_uuid": self._parent_uuid,
        }

    # ── messaging ──────────────────────────────────────────────────────────

    async def send_message(
        self,
        prompt: str,
        files: list[str | Path] | None = None,
        model: str | None = None,
    ) -> ModelOutput:
        """
        Send *prompt* and wait for the complete response.

        Parameters
        ----------
        prompt:
            The user message to send.
        files:
            Optional list of local file paths to attach.
        model:
            Override the session-level model for this turn only.

        Returns
        -------
        ModelOutput
        """
        output = await self._client._send(
            conv_id       = self._conv_id,
            prompt        = prompt,
            files         = files,
            model         = model or self._model,
            parent_uuid   = self._parent_uuid,
            system_prompt = self._system_prompt,
            new_conv      = self._last_response is None,
        )
        self._last_response = output
        if output.metadata.get("parent_message_uuid"):
            self._parent_uuid = output.metadata["parent_message_uuid"]
        return output

    async def send_message_stream(
        self,
        prompt: str,
        files: list[str | Path] | None = None,
        model: str | None = None,
    ) -> AsyncIterator[ModelOutput]:
        """
        Stream *prompt* response, yielding incremental :class:`ModelOutput`
        chunks.  The ``text_delta`` attribute of each chunk contains only the
        new text since the previous chunk.

        Example::

            async for chunk in chat.send_message_stream("Write me an essay"):
                print(chunk.text_delta, end="", flush=True)
        """
        async for chunk in self._client._send_stream(
            conv_id       = self._conv_id,
            prompt        = prompt,
            files         = files,
            model         = model or self._model,
            parent_uuid   = self._parent_uuid,
            system_prompt = self._system_prompt,
            new_conv      = self._last_response is None,
        ):
            # Update parent UUID from the last chunk's metadata
            if chunk.metadata.get("parent_message_uuid"):
                self._parent_uuid = chunk.metadata["parent_message_uuid"]
            yield chunk

        self._last_response = ModelOutput(text="<streamed>")

    # ── candidate selection ────────────────────────────────────────────────

    def choose_candidate(self, index: int) -> None:
        """
        Select which reply candidate should be used as context for the
        *next* message in this conversation.

        Parameters
        ----------
        index:
            Zero-based index into :attr:`ModelOutput.candidates`.
        """
        if self._last_response is None:
            raise RuntimeError("No response to choose from yet.")
        candidates = self._last_response.candidates
        if not 0 <= index < len(candidates):
            raise IndexError(
                f"Candidate index {index} out of range "
                f"(0–{len(candidates) - 1})."
            )
        self._current_candidate_index = index

    # ── delete ─────────────────────────────────────────────────────────────

    async def delete(self) -> None:
        """Delete this conversation from Claude.ai history."""
        await self._client.delete_conversation(self._conv_id)

