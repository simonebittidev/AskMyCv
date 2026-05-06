"""Abstract base class for ingestion sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from kg_builder.models import SourceDocument


class DocumentSource(ABC):
    """A pluggable provider of :class:`SourceDocument` instances.

    Implementations must be stateless or own their own state internally — the
    pipeline calls :meth:`load` once per build.
    """

    @abstractmethod
    async def load(self) -> List[SourceDocument]:
        """Return the documents this source produces."""
