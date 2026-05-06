"""Wrap LangChain's ``LLMGraphTransformer`` with per-source schema control.

We keep ``LLMGraphTransformer`` (well-supported, Neo4j-compatible) but apply
strict per-source schemas so that:

* CV / Cover Letter cannot produce ``PersonalProject`` / ``Project`` nodes.
* GitHub READMEs produce ``PersonalProject`` and tightly-scoped tech entities.

This is the practical 2026 stance: deterministic schema constraints beat free-
form extraction for downstream retrieval quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

from kg_builder.extraction._llm import get_chat_llm
from kg_builder.graph.schema import SourceSchema

logger = logging.getLogger(__name__)


@dataclass
class GraphExtractionResult:
    graph_documents: list  # list[GraphDocument] — keep loose to avoid cross-version import friction


class GraphExtractor:
    """Convert LangChain documents into graph documents under a constrained schema."""

    def __init__(self, schema: SourceSchema, additional_instructions: str = "") -> None:
        self._schema = schema
        self._extra = additional_instructions
        self._transformer = LLMGraphTransformer(
            llm=get_chat_llm(temperature=0.0),
            allowed_nodes=list(schema.allowed_nodes),
            allowed_relationships=list(schema.allowed_relationships) or None,
            strict_mode=schema.strict,
            additional_instructions=self._build_instructions(),
        )

    def _build_instructions(self) -> str:
        parts = [self._schema.instructions, self._extra]
        return "\n\n".join(p for p in parts if p)

    async def extract(self, documents: Sequence[Document]) -> GraphExtractionResult:
        if not documents:
            return GraphExtractionResult(graph_documents=[])
        graph_docs = await self._transformer.aconvert_to_graph_documents(list(documents))
        logger.info(
            "[graph-extractor] %s docs -> %s graph documents",
            len(documents),
            len(graph_docs),
        )
        return GraphExtractionResult(graph_documents=graph_docs)
