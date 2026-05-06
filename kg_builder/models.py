"""Pydantic models used across the kg_builder pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# --- Source documents --------------------------------------------------------


class SourceDocument(BaseModel):
    """Generic ingested document, before any LLM processing."""

    content: str = Field(description="The document content as Markdown text.")
    source_id: str = Field(description="Stable identifier (filename, repo full name, ...).")
    kind: str = Field(description="Document kind: 'cv', 'cover_letter', 'github_readme'.")
    metadata: dict = Field(default_factory=dict, description="Arbitrary source metadata.")


# --- Summarisation & chunking -----------------------------------------------


class DocumentSummary(BaseModel):
    brief_overview: str = Field(
        description="A brief summary (2-3 sentences) capturing the document's purpose."
    )
    detailed_summary: str = Field(
        description="An exhaustive summary preserving every relevant fact about the person."
    )


class Chunk(BaseModel):
    title: str = Field(description="A short, descriptive title for this chunk.")
    content: str = Field(description="The text content of this chunk.")
    keywords: List[str] = Field(description="Uppercase keywords summarising the chunk.")


class ChunkedSummary(BaseModel):
    chunks: List[Chunk] = Field(description="Ordered list of semantically meaningful chunks.")


# --- GitHub projects ---------------------------------------------------------


class ProjectDoc(BaseModel):
    """Normalised metadata + README for a GitHub project."""

    repo: str = Field(description="Full repo name 'owner/name'.")
    name: str = Field(description="Repo name (no owner).")
    display_name: Optional[str] = Field(default=None)
    url: str = Field(description="HTML URL of the repo.")
    description: Optional[str] = Field(default=None)
    homepage: Optional[str] = Field(default=None)
    primary_language: Optional[str] = Field(default=None)
    topics: List[str] = Field(default_factory=list)
    stars: int = Field(default=0)
    pushed_at: Optional[datetime] = Field(default=None)
    readme_md: str = Field(description="README content as Markdown.")
