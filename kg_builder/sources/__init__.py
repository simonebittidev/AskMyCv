"""Data sources that produce :class:`~kg_builder.models.SourceDocument` instances."""

from kg_builder.sources.base import DocumentSource
from kg_builder.sources.github_source import GitHubSource
from kg_builder.sources.pdf_source import PdfSource

__all__ = ["DocumentSource", "GitHubSource", "PdfSource"]
