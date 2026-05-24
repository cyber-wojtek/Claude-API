"""
Output data models for claude_webapi.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import aiohttp


# ──────────────────────────────────────────────────────────────────────────────
# Image
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Image:
    """A single image that appeared in a Claude response."""

    url: str
    alt: str = ""
    title: str = ""
    # True when Claude generated the image; False when fetched from the web
    generated: bool = False

    def __repr__(self) -> str:
        kind = "GeneratedImage" if self.generated else "WebImage"
        return f"<{kind} title={self.title!r} alt={self.alt!r} url={self.url!r}>"

    async def save(
        self,
        path: str | Path = ".",
        filename: str | None = None,
        verbose: bool = False,
    ) -> Path:
        """
        Download and save the image to *path/filename*.

        Parameters
        ----------
        path:
            Directory to save into (created if it does not exist).
        filename:
            Override the file name.  Defaults to the last segment of the URL.
        verbose:
            Print save path on success.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        dest_dir = Path(path)
        dest_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            filename = self.url.split("/")[-1].split("?")[0] or "image.png"

        dest = dest_dir / filename

        async with aiohttp.ClientSession() as session:
            async with session.get(self.url) as resp:
                resp.raise_for_status()
                dest.write_bytes(await resp.read())

        if verbose:
            print(f"Saved: {dest}")
        return dest.resolve()


# ──────────────────────────────────────────────────────────────────────────────
# Candidate
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """One reply candidate in a ModelOutput."""

    index: int
    text: str
    images: list[Image] = field(default_factory=list)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"<Candidate [{self.index}] {preview!r}>"


# ──────────────────────────────────────────────────────────────────────────────
# ModelOutput
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelOutput:
    """
    The structured result of a single Claude generation.

    Attributes
    ----------
    text:
        The full text of the primary (first) candidate.
    candidates:
        All reply candidates returned by Claude.
    images:
        Images found in the primary candidate.
    thoughts:
        Any extended-thinking / reasoning text surfaced by the model.
    metadata:
        Raw conversation metadata (used for session continuation).
    text_delta:
        In streaming mode, contains only the new text received since the
        last yielded chunk.  Empty string for non-streaming outputs.
    """

    text: str
    candidates: list[Candidate] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)
    thoughts: str = ""
    metadata: dict = field(default_factory=dict)
    text_delta: str = ""
    thinking_delta:  str = ""
    
    # ── tool use ──────────────────────────────────────────────────────────
    tool_index:      int | None      = None   # which block index this belongs to
    tool_id:         str             = ""     # tool use id from message
    tool_name:       str             = ""     # e.g. "artifacts", "web_search"
    tool_input_delta: str            = ""     # partial JSON input delta
    tool_result:     dict | None     = None   # complete tool_result block if present
    event_type:      str             = ""     # raw SSE event type for passthrough
    raw_event:       dict | None     = None   # full raw SSE event for passthrough
    tool_use_id:     str             = ""     # unique ID for this tool use, for correlating across events
    
    # ── convenience ───────────────────────────────────────────────────────

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        preview = self.text[:120].replace("\n", " ")
        return f"<ModelOutput text={preview!r}>"

    # ── image helpers ──────────────────────────────────────────────────────

    @property
    def web_images(self) -> list[Image]:
        """Web-sourced images only."""
        return [img for img in self.images if not img.generated]

    @property
    def generated_images(self) -> list[Image]:
        """AI-generated images only."""
        return [img for img in self.images if img.generated]
    
    @property
    def all_images(self) -> list[Image]:
        """All images, regardless of source."""
        return self.images
    
    # ── candidate helpers ─────────────────────────────────────────────────
    @property
    def primary_candidate(self) -> Candidate | None:
        """The primary (first) candidate, or None if no candidates."""
        return self.candidates[0] if self.candidates else None
    
    @property
    def primary_text(self) -> str:
        """The text of the primary candidate, or empty string if no candidates."""
        return self.primary_candidate.text if self.primary_candidate else ""
    
    @property
    def primary_images(self) -> list[Image]:
        """The images of the primary candidate, or empty list if no candidates."""
        return self.primary_candidate.images if self.primary_candidate else []
    
    # ── internal ──────────────────────────────────────────────────────────
    @classmethod
    def from_raw_event(cls, event: dict) -> ModelOutput:
        """Parse a raw SSE event dict into a ModelOutput."""
        text = event.get("text", "")
        images = _extract_images(text)
        candidates = []
        for i, cand_text in enumerate(event.get("candidates", [])):
            cand_images = _extract_images(cand_text)
            candidates.append(Candidate(index=i, text=cand_text, images=cand_images))
        thoughts = event.get("thoughts", "")
        metadata = event.get("metadata", {})
        return cls(
            text=text,
            candidates=candidates,
            images=images,
            thoughts=thoughts,
            metadata=metadata,
            text_delta=event.get("text_delta", ""),
            thinking_delta=event.get("thinking_delta", ""),
            tool_index=event.get("tool_index"),
            tool_id=event.get("tool_id", ""),
            tool_name=event.get("tool_name", ""),
            tool_input_delta=event.get("tool_input_delta", ""),
            tool_result=event.get("tool_result"),
            event_type=event.get("event_type", ""),
            raw_event=event
        )
    
    @classmethod
    def from_final_response(cls, response: dict) -> ModelOutput:
        """Parse the final JSON response from a non-streaming send_message into a ModelOutput."""
        text = response.get("text", "")
        images = _extract_images(text)
        candidates = []
        for i, cand_text in enumerate(response.get("candidates", [])):
            cand_images = _extract_images(cand_text)
            candidates.append(Candidate(index=i, text=cand_text, images=cand_images))
        thoughts = response.get("thoughts", "")
        metadata = response.get("metadata", {})
        return cls(
            text=text,
            candidates=candidates,
            images=images,
            thoughts=thoughts,
            metadata=metadata
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — parse SSE stream into ModelOutput
# ──────────────────────────────────────────────────────────────────────────────

_IMG_MD_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^\)]+)\)")


def _extract_images(text: str) -> list[Image]:
    """Return Image objects for every markdown image in *text*."""
    imgs = []
    for alt, url in _IMG_MD_RE.findall(text):
        imgs.append(Image(url=url, alt=alt))
    return imgs

