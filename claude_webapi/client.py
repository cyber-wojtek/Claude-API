"""
ClaudeClient — async client for the Claude.ai web API.
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import quote

import aiohttp

from .constants import DEFAULT_MODEL, CLAUDE_BASE_URL, Model
from .exceptions import (
    APIError, AuthenticationError, ConversationNotFoundError,
    FileUploadError, QuotaExceededError, TimeoutError,
)
from .session import ChatSession
from .types import Candidate, Image, ModelOutput, _extract_images

logger = logging.getLogger("claude_webapi")


# ──────────────────────────────────────────────────────────────────────────────
# Default tools payload (mirrors what Claude.ai web sends)
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_TOOLS: list[dict] = [
    {"name": "web_search",            "type": "web_search_v0"},
    {"name": "artifacts",             "type": "artifacts_v0"},
    {"name": "repl",                  "type": "repl_v0"},
    {"name": "ask_user_input_v0",     "type": "widget"},
    {"name": "weather_fetch",         "type": "widget"},
    {"name": "recipe_display_v0",     "type": "widget"},
    {"name": "places_map_display_v0", "type": "widget"},
    {"name": "message_compose_v1",    "type": "widget"},
    {"name": "places_search",         "type": "widget"},
    {"name": "fetch_sports_data",     "type": "widget"},
]

_DEFAULT_STYLE: dict = {
    "isDefault": True, "key": "default", "name": "Normal",
    "nameKey": "normal_style_name", "prompt": "Normal\n",
    "summary": "Default responses from Claude",
    "summaryKey": "normal_style_summary", "type": "default",
}

_COMMON_HEADERS: dict[str, str] = {
    "Accept-Language":             "en-US,en;q=0.9",
    "Alt-Used":                    "claude.ai",
    "anthropic-client-platform":  "web_claude_ai",
    "anthropic-client-version":   "1.0.0",
    "Connection":                  "keep-alive",
    "Origin":                      "https://claude.ai",
    "Referer":                     "https://claude.ai/new",
    "Sec-Fetch-Dest":              "empty",
    "Sec-Fetch-Mode":              "cors",
    "Sec-Fetch-Site":              "same-origin",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:147.0) "
        "Gecko/20100101 Firefox/147.0"
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# ClaudeClient
# ──────────────────────────────────────────────────────────────────────────────

class ClaudeClient:
    """
    Async client for the Claude.ai web API.

    Parameters
    ----------
    session_key:
        The ``sessionKey`` cookie value from claude.ai.
    organization_id:
        Your Claude organization UUID (visible in the ``lastActiveOrg``
        cookie or the URL after signing in).
    proxy:
        Optional HTTP/S proxy URL, e.g. ``"http://user:pass@host:port"``.

    Example::

        import asyncio
        from claude_webapi import ClaudeClient

        async def main():
            client = ClaudeClient("sk-ant-…", "xxxxxxxx-…")
            await client.init()
            response = await client.generate_content("Hello!")
            print(response.text)

        asyncio.run(main())
    """

    def __init__(
        self,
        session_key: str,
        organization_id: str,
        proxy: str | None = None,
    ):
        if not session_key:
            raise AuthenticationError("session_key must not be empty.")
        if not organization_id:
            raise AuthenticationError("organization_id must not be empty.")

        self._session_key     = session_key
        self._organization_id = organization_id
        self._proxy           = proxy
        self._session: aiohttp.ClientSession | None = None
        self._auto_close      = False
        self._close_delay     = 300
        self._close_task: asyncio.Task | None = None


    # ── lifecycle ──────────────────────────────────────────────────────────

    async def init(
        self,
        timeout: int = 30,
        auto_close: bool = False,
        close_delay: int = 300,
    ) -> None:
        """
        Initialise the underlying HTTP session and verify credentials.

        Parameters
        ----------
        timeout:
            Default request timeout in seconds.
        auto_close:
            Automatically close the session after *close_delay* seconds of
            inactivity.  Useful for long-running services.
        close_delay:
            Inactivity seconds before auto-close (requires *auto_close=True*).
        """
        self._auto_close  = auto_close
        self._close_delay = close_delay
        self._timeout     = aiohttp.ClientTimeout(total=timeout)

        cookies = {
            "sessionKey":    self._session_key,
            "lastActiveOrg": self._organization_id,
        }

        connector = aiohttp.TCPConnector(ssl=True)
        self._session = aiohttp.ClientSession(
            cookies=cookies,
            connector=connector,
            headers=_COMMON_HEADERS,
        )
        logger.info("ClaudeClient initialised (org=%s…)", self._organization_id[:8])

    async def close(self) -> None:
        """Explicitly close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("ClaudeClient session closed.")

    async def __aenter__(self) -> "ClaudeClient":
        await self.init()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    def _reset_close_timer(self) -> None:
        if not self._auto_close:
            return
        if self._close_task and not self._close_task.done():
            self._close_task.cancel()
        self._close_task = asyncio.create_task(self._delayed_close())

    async def _delayed_close(self) -> None:
        await asyncio.sleep(self._close_delay)
        logger.info("Auto-closing session after %ds inactivity.", self._close_delay)
        await self.close()

    # ── URL helpers ────────────────────────────────────────────────────────

    def _org_url(self, path: str) -> str:
        return f"{CLAUDE_BASE_URL}/api/organizations/{self._organization_id}/{path}"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            raise RuntimeError(
                "Client not initialised.  Call `await client.init()` first."
            )
        return self._session

    # ── low-level request helpers ──────────────────────────────────────────

    async def _get(self, url: str, **kwargs) -> dict:
        session = self._ensure_session()
        self._reset_close_timer()
        async with session.get(url, timeout=self._timeout, **kwargs) as resp:
            await self._raise_for_status(resp)
            return await resp.json(content_type=None)

    async def _post(self, url: str, payload: dict, **kwargs) -> dict:
        session = self._ensure_session()
        self._reset_close_timer()
        body = json.dumps(payload).encode()
        async with session.post(
            url,
            data=body,
            headers={"Content-Length": str(len(body)), "content-type": "application/json"},
            timeout=self._timeout,
            **kwargs,
        ) as resp:
            await self._raise_for_status(resp)
            return await resp.json(content_type=None)

    async def _put(self, url: str, payload: dict) -> dict:
        session = self._ensure_session()
        self._reset_close_timer()
        body = json.dumps(payload).encode()
        async with session.put(
            url,
            data=body,
            headers={"Content-Length": str(len(body)), "content-type": "application/json"},
            timeout=self._timeout,
        ) as resp:
            await self._raise_for_status(resp)
            return await resp.json(content_type=None)

    @staticmethod
    async def _raise_for_status(resp: aiohttp.ClientResponse) -> None:
        if resp.status == 401:
            raise AuthenticationError("Invalid or expired sessionKey.")
        if resp.status == 404:
            raise ConversationNotFoundError(
                f"Resource not found: {resp.url}"
            )
        if resp.status >= 400:
            body = await resp.text()
            raise APIError(
                f"HTTP {resp.status}: {body[:400]}", status_code=resp.status
            )

    # ── conversation management ────────────────────────────────────────────

    async def list_conversations(self) -> list[dict]:
        """Return all conversations for the current organisation."""
        return await self._get(self._org_url("chat_conversations"))

    async def get_conversation(self, conversation_id: str) -> dict:
        """Fetch full details of a single conversation."""
        path = (
            f"chat_conversations/{conversation_id}"
            "?tree=True&rendering_mode=messages"
            "&render_all_tools=true&consistency=strong"
        )
        return await self._get(self._org_url(path))

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete *conversation_id* from Claude.ai history."""
        session = self._ensure_session()
        url = self._org_url(f"chat_conversations/{conversation_id}")
        async with session.delete(url, timeout=self._timeout) as resp:
            if resp.status not in (200, 204):
                body = await resp.text()
                raise APIError(f"Delete failed ({resp.status}): {body}")
        logger.info("Deleted conversation %s…", conversation_id[:8])

    async def update_conversation_settings(self, conversation_id: str, settings: dict) -> None:
        """Update conversation settings."""
        await self._put(
            self._org_url(f"chat_conversations/{conversation_id}"),
            payload=settings,
        )

    # ── file operations ────────────────────────────────────────────────────

    async def upload_file(
        self,
        conversation_id: str,
        file_path: str | Path,
    ) -> str:
        """
        Upload a local file to an existing conversation.

        Returns
        -------
        str
            The ``file_uuid`` assigned by Claude.ai.
        """
        path   = Path(file_path)
        mime   = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data   = path.read_bytes()
        url    = (
            f"{CLAUDE_BASE_URL}/api/organizations/{self._organization_id}"
            f"/conversations/{conversation_id}/wiggle/upload-file"
        )
        session = self._ensure_session()
        form   = aiohttp.FormData()
        form.add_field("file", data, filename=path.name, content_type=mime)

        # No content-type header needed — aiohttp.FormData sets its own
        # multipart/form-data boundary automatically.
        async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=90)) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise FileUploadError(
                    f"Upload failed ({resp.status}): {body[:300]}"
                )
            result = await resp.json(content_type=None)
            fid = result.get("file_uuid") or result.get("id", "")
            logger.info("Uploaded %s → %s…", path.name, str(fid)[:8])
            return fid

    async def download_file(
        self,
        conversation_id: str,
        file_path: str,
        dest: str | Path = ".",
    ) -> Path:
        """
        Download a file from a conversation's sandbox.

        Parameters
        ----------
        file_path:
            Server-side path of the file to download.
        dest:
            Local directory to save into.
        """
        url = (
            f"{CLAUDE_BASE_URL}/api/organizations/{self._organization_id}"
            f"/conversations/{conversation_id}/wiggle/download-file"
            f"?path={quote(file_path, safe='')}"
        )
        session = self._ensure_session()
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=90)) as resp:
            await self._raise_for_status(resp)
            filename = file_path.split("/")[-1] or "download"
            out = Path(dest) / filename
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(await resp.read())
            return out.resolve()

    # ── internal: ensure conversation exists ──────────────────────────────

    async def _ensure_conversation(self, conv_id: str) -> None:
        """Create the conversation server-side if it doesn't yet exist."""
        payload = {
            "include_conversation_preferences": True,
            "is_temporary": False,
            "name": "",
            "uuid": conv_id,
        }
        session = self._ensure_session()
        body    = json.dumps(payload).encode()
        async with session.post(
            self._org_url("chat_conversations"),
            data=body,
            headers={"Content-Length": str(len(body)), "content-type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status in (200, 201):
                return
            # Claude.ai returns 400 or 409 when conversation already exists
            # — that's fine for an "ensure" operation.
            if resp.status in (400, 409):
                body_text = await resp.text()
                if "could not be created" in body_text.lower() or resp.status == 409:
                    logger.debug("Conversation %s already exists, continuing.", conv_id[:8])
                    return
                raise APIError(
                    f"Could not create conversation ({resp.status}): {body_text[:300]}"
                )
            body_text = await resp.text()
            raise APIError(
                f"Could not create conversation ({resp.status}): {body_text[:300]}"
            )

    # ── internal: upload files listed as paths ────────────────────────────

    async def _upload_file_list(
        self, conv_id: str, files: list[str | Path]
    ) -> list[str]:
        """Upload any items in *files* that are local paths; return UUIDs."""
        uuids = []
        for f in files:
            p = Path(f)
            if p.exists():
                fid = await self.upload_file(conv_id, p)
                uuids.append(fid)
            else:
                # Assume already a UUID string
                uuids.append(str(f))
        return uuids

    # ── internal: build completion payload ────────────────────────────────

    @staticmethod
    def _build_payload(
        prompt: str,
        file_uuids: list[str],
        model: str,
        parent_uuid: str,
        system_prompt: str | None,
    ) -> dict:
        tools = list(_DEFAULT_TOOLS)
        payload: dict = {
            "files":               file_uuids,
            "locale":              "en-US",
            "model":               model,
            "parent_message_uuid": parent_uuid,
            "personalized_styles": [_DEFAULT_STYLE],
            "prompt":              prompt,
            "rendering_mode":      "messages",
            "sync_sources":        [],
            "timezone":            "UTC",
            "tools":               tools,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        return payload

    # ── internal: parse SSE stream ─────────────────────────────────────────

    @staticmethod
    def _parse_sse_chunk(raw: bytes) -> dict | None:
        """Parse a single SSE data line into a dict, or None."""
        try:
            text = raw.decode("utf-8", errors="replace").strip()
        except Exception:
            return None
        for line in text.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    try:
                        return json.loads(payload)
                    except json.JSONDecodeError:
                        pass
        return None

    # ── internal: send (non-streaming) ────────────────────────────────────

    async def _send(
        self,
        conv_id: str,
        prompt: str,
        files: list[str | Path] | None,
        model: str | None,
        parent_uuid: str,
        system_prompt: str | None,
        new_conv: bool,
    ) -> ModelOutput:
        if new_conv:
            await self._ensure_conversation(conv_id)

        file_uuids = await self._upload_file_list(conv_id, files or [])
        resolved_model = _resolve_model(model)
        payload = self._build_payload(
            prompt, file_uuids, resolved_model, parent_uuid, system_prompt
        )

        url     = self._org_url(f"chat_conversations/{conv_id}/completion")
        session = self._ensure_session()
        body    = json.dumps(payload).encode()

        full_text    = ""
        thoughts     = ""
        new_parent   = parent_uuid
        meta: dict   = {}

        async with session.post(
            url,
            data=body,
            headers={
                "Accept":         "text/event-stream",
                "Content-Length": str(len(body)),
                "content-type":   "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=3600),
        ) as resp:
            await self._raise_for_status(resp)
            async for raw_chunk, _ in resp.content.iter_chunks():
                evt = self._parse_sse_chunk(raw_chunk)
                if not evt:
                    continue
                etype = evt.get("type", "")

                if etype == "content_block_delta":
                    delta = evt.get("delta", {})
                    if delta.get("type") == "text_delta":
                        full_text += delta.get("text", "")
                    elif delta.get("type") == "thinking_delta":
                        thoughts += delta.get("thinking", "")

                elif etype == "message_stop":
                    meta = evt.get("message", {})
                    mid  = meta.get("uuid") or meta.get("id", "")
                    if mid:
                        new_parent = mid

                elif etype == "message_limit":
                    raise QuotaExceededError(
                        "Claude.ai message limit reached. "
                        "Try again later or switch accounts."
                    )

        images = _extract_images(full_text)
        candidate = Candidate(index=0, text=full_text, images=images)
        return ModelOutput(
            text       = full_text,
            candidates = [candidate],
            images     = images,
            thoughts   = thoughts,
            metadata   = {"parent_message_uuid": new_parent, **meta},
        )

    async def _stop(self, conversation_id: str) -> bool:
        """Send a stop signal to the server to halt an in-progress response."""
        url = (
            f"{CLAUDE_BASE_URL}/api/organizations/{self._organization_id}"
            f"/chat_conversations/{conversation_id}/stop_response"
        )
        session = self._ensure_session()
        async with session.post(url, data=b"", timeout=aiohttp.ClientTimeout(total=10)) as resp:
            return resp.status == 200

    # ── internal: send (streaming) ─────────────────────────────────────────

    async def _send_stream(
        self,
        conv_id: str,
        prompt: str,
        files: list[str | Path] | None,
        model: str | None,
        parent_uuid: str,
        system_prompt: str | None,
        new_conv: bool,
    ) -> AsyncIterator[ModelOutput]:
        if new_conv:
            await self._ensure_conversation(conv_id)

        file_uuids = await self._upload_file_list(conv_id, files or [])
        resolved_model = _resolve_model(model)
        payload = self._build_payload(
            prompt, file_uuids, resolved_model, parent_uuid, system_prompt
        )

        url     = self._org_url(f"chat_conversations/{conv_id}/completion")
        session = self._ensure_session()
        body    = json.dumps(payload).encode()

        accumulated = ""
        new_parent  = parent_uuid
        meta: dict  = {}

        async with session.post(
            url,
            data=body,
            headers={
                "Accept":         "text/event-stream",
                "Content-Length": str(len(body)),
                "content-type":   "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            await self._raise_for_status(resp)
            async for raw_chunk, _ in resp.content.iter_chunks():
                evt = self._parse_sse_chunk(raw_chunk)
                if not evt:
                    continue
                etype = evt.get("type", "")

                if etype == "content_block_delta":
                    delta = evt.get("delta", {})
                    if delta.get("type") == "text_delta":
                        delta_text  = delta.get("text", "")
                        accumulated += delta_text
                        yield ModelOutput(
                            text       = accumulated,
                            text_delta = delta_text,
                            metadata   = {"parent_message_uuid": new_parent},
                        )

                elif etype == "message_stop":
                    meta = evt.get("message", {})
                    mid  = meta.get("uuid") or meta.get("id", "")
                    if mid:
                        new_parent = mid

                elif etype == "message_limit":
                    raise QuotaExceededError(
                        "Claude.ai message limit reached."
                    )

    # ── public API: generate_content ──────────────────────────────────────

    async def generate_content(
        self,
        prompt: str,
        files: list[str | Path] | None = None,
        model: str | Model | None = None,
        system_prompt: str | None = None,
    ) -> ModelOutput:
        """
        Send a single-turn message to Claude and return the full response.

        Parameters
        ----------
        prompt:
            The user message.
        files:
            Optional list of local file paths to attach.
        model:
            Model to use.  Accepts a :class:`~claude_webapi.constants.Model`
            enum member or a raw model-string.  Defaults to
            ``claude-sonnet-4-6``.
        system_prompt:
            Optional system prompt prepended to this single generation.

        Returns
        -------
        ModelOutput

        Example::

            response = await client.generate_content("What is 2 + 2?")
            print(response.text)
        """
        conv_id = str(uuid.uuid4())
        await self._ensure_conversation(conv_id)
        return await self._send(
            conv_id       = conv_id,
            prompt        = prompt,
            files         = files,
            model         = model,
            parent_uuid   = "00000000-0000-4000-8000-000000000000",
            system_prompt = system_prompt,
            new_conv      = False,  # already created above
        )

    # ── public API: generate_content_stream ───────────────────────────────

    async def generate_content_stream(
        self,
        prompt: str,
        files: list[str | Path] | None = None,
        model: str | Model | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[ModelOutput]:
        """
        Stream a single-turn response, yielding incremental chunks.

        Each yielded :class:`ModelOutput` has a ``text_delta`` attribute with
        only the new text received since the previous chunk.

        Example::

            async for chunk in client.generate_content_stream("Tell me a story"):
                print(chunk.text_delta, end="", flush=True)
        """
        conv_id = str(uuid.uuid4())
        await self._ensure_conversation(conv_id)
        async for chunk in self._send_stream(
            conv_id       = conv_id,
            prompt        = prompt,
            files         = files,
            model         = model,
            parent_uuid   = "00000000-0000-4000-8000-000000000000",
            system_prompt = system_prompt,
            new_conv      = False,
        ):
            yield chunk
            
    async def stop_response(self, conversation_id: str) -> bool:
        """
        Stop an in-progress response for *conversation_id*.

        Parameters
        ----------
        conversation_id:
            UUID of the conversation to stop.
        Returns
        -------
        bool            
            
            
        True if the stop signal was successfully sent, False otherwise.
        """        
        return await self._stop(conversation_id)

    # ── public API: start_chat ────────────────────────────────────────────

    def start_chat(
        self,
        model: str | Model | None = None,
        system_prompt: str | None = None,
        metadata: dict | None = None,
    ) -> ChatSession:
        """
        Create a new :class:`~claude_webapi.session.ChatSession`.

        Parameters
        ----------
        model:
            Model to use for the session.
        system_prompt:
            A persistent system prompt for the entire conversation.
        metadata:
            Pass a previously saved ``chat.metadata`` dict to resume a
            conversation from a prior session.

        Returns
        -------
        ChatSession

        Example::

            chat = client.start_chat()
            r1 = await chat.send_message("Hello!")
            r2 = await chat.send_message("What did I just say?")
        """
        resolved = _resolve_model(model) if model else None
        return ChatSession(
            client        = self,
            model         = resolved,
            system_prompt = system_prompt,
            metadata      = metadata,
        )

    # ── public API: delete_conversation ───────────────────────────────────

    async def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete *conversation_id* from Claude.ai's history.

        Parameters
        ----------
        conversation_id:
            UUID of the conversation to delete.

        Example::

            await client.delete_conversation(chat.cid)
        """
        session = self._ensure_session()
        url     = self._org_url(f"chat_conversations/{conversation_id}")
        async with session.delete(url, timeout=self._timeout) as resp:
            if resp.status not in (200, 204):
                body = await resp.text()
                raise APIError(f"Delete failed ({resp.status}): {body}")
        logger.info("Deleted conversation %s…", conversation_id[:8])

    # ── account settings ───────────────────────────────────────────────────

    async def patch_settings(self, payload: dict) -> None:
        """Apply *payload* to the account's Claude.ai settings."""
        session = self._ensure_session()
        body    = json.dumps(payload).encode()
        async with session.patch(
            f"{CLAUDE_BASE_URL}/api/account/settings",
            data=body,
            headers={"Content-Length": str(len(body)), "content-type": "application/json"},
            timeout=self._timeout,
        ) as resp:
            await self._raise_for_status(resp)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_model(model: str | Model | None) -> str:
    """Normalise a model argument to a plain string."""
    if model is None:
        return DEFAULT_MODEL
    if isinstance(model, Model):
        return model.value
    return str(model)

