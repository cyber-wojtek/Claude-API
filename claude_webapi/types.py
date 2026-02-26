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

