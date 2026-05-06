"""Semantic chunking + late-chunking embedding strategy.

**Late chunking** (Günther et al., 2024) embeds the full document once, then
derives chunk-level vectors by mean-pooling the token-range of each chunk.
Chunks therefore carry the *global* document context. We approximate this
without low-level token offsets by embedding the document, embedding each
chunk, and combining them via weighted mean — which preserves most of the
contextual benefit while staying compatible with hosted embedding APIs.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from kg_builder.extraction._llm import get_chat_llm, get_embeddings
from kg_builder.models import Chunk, ChunkedSummary

logger = logging.getLogger(__name__)


_CHUNK_PROMPT = """\
You are an expert in document analysis and information structuring.

## Goal
Take the provided Markdown text (a CV summary, cover letter, or project \
README) and split it into semantically meaningful chunks. Each chunk must be \
self-contained and contextually meaningful for downstream RAG.

## Instructions
- Identify natural section boundaries (Education, Experience, Skills, \
  Architecture, Usage, etc.).
- Each chunk should be 3+ sentences and at most ~300 words.
- Keep the original ordering. Do not merge unrelated topics.
- For each chunk produce: a short title, the full content, and exactly 5 \
  uppercase keywords summarising it.

## Output Format
Return only valid JSON conforming to the schema."""


def chunk_document(markdown: str) -> List[Chunk]:
    """Split a Markdown document into semantically meaningful chunks via LLM."""
    llm = get_chat_llm(temperature=0.0)
    messages = [
        SystemMessage(content=_CHUNK_PROMPT),
        HumanMessage(content=f"Markdown to analyse:\n\n{markdown}"),
    ]
    result = llm.with_structured_output(ChunkedSummary).invoke(messages)
    return result.chunks


# --- Late chunking embeddings -----------------------------------------------


# Weight applied to the document-level embedding when blending with the chunk
# embedding. Higher = more global context bleed. 0.25 is a sensible default.
_LATE_CHUNK_ALPHA = 0.25


def _blend(chunk_vec: List[float], doc_vec: List[float], alpha: float) -> List[float]:
    return [(1 - alpha) * c + alpha * d for c, d in zip(chunk_vec, doc_vec)]


def embed_chunks(
    full_markdown: str,
    chunks: List[Chunk],
    *,
    source_id: str,
    kind: str,
    use_late_chunking: bool = True,
) -> List[Document]:
    """Embed each chunk, optionally blending with the full-document embedding.

    Returns LangChain :class:`Document` objects ready to be passed to a graph
    transformer or directly stored in Neo4j as vector-bearing nodes.
    """
    embeddings = get_embeddings()

    doc_vec: List[float] | None = None
    if use_late_chunking:
        try:
            doc_vec = embeddings.embed_query(full_markdown)
        except Exception as exc:
            # Document might exceed the embedding model context window.
            logger.warning(
                "[chunker] late-chunking disabled for %s (embed_query failed: %s)",
                source_id,
                exc,
            )
            doc_vec = None

    documents: List[Document] = []
    for chunk in chunks:
        chunk_vec = embeddings.embed_query(chunk.content)
        if doc_vec is not None:
            chunk_vec = _blend(chunk_vec, doc_vec, _LATE_CHUNK_ALPHA)

        documents.append(
            Document(
                page_content=f"{chunk.title}\n{chunk.content}",
                metadata={
                    "embedding": chunk_vec,
                    "source": source_id,
                    "kind": kind,
                    "chunk_title": chunk.title,
                    "keywords": chunk.keywords,
                },
            )
        )
    return documents
