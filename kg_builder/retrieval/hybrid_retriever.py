"""Hybrid retriever: vector + fulltext + graph traversal + community summaries.

This is the 2026 retrieval pattern: a single vector match expanded into the
graph neighbourhood, optionally boosted with project-aware routing and
community-level "global search" answers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from langchain_neo4j import Neo4jVector

from kg_builder.extraction._llm import get_embeddings
from kg_builder.graph.neo4j_client import Neo4jClient, get_default_client

logger = logging.getLogger(__name__)


_PROJECT_HINTS_RE = re.compile(
    r"\b(project|projects|repo|repos|repository|github|portfolio|side\s*project|personal\s*project)\b",
    re.IGNORECASE,
)


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source: str = ""

    def to_string(self) -> str:
        return self.text


class HybridGraphRetriever:
    """Vector match on Document nodes + 1-hop graph context + project boost."""

    def __init__(
        self,
        client: Optional[Neo4jClient] = None,
        *,
        top_k: int = 6,
        graph_hop_limit: int = 1,
    ) -> None:
        self._client = client or get_default_client()
        self._top_k = top_k
        self._hop = graph_hop_limit

    # ------------------------------------------------------------------

    def _vector_index(self) -> Neo4jVector:
        return Neo4jVector.from_existing_graph(
            embedding=get_embeddings(),
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )

    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        chunks: List[RetrievedChunk] = []

        # 1. Vector / fulltext hybrid search on chunk Documents.
        try:
            for doc, score in self._vector_index().similarity_search_with_score(
                question, k=self._top_k
            ):
                chunks.append(
                    RetrievedChunk(
                        text=doc.page_content,
                        score=float(score),
                        source=doc.metadata.get("source", ""),
                    )
                )
        except Exception as exc:
            logger.warning("[retriever] vector search failed: %s", exc)

        # 2. If the question is project-flavoured, surface PersonalProject context.
        if _PROJECT_HINTS_RE.search(question):
            chunks.extend(self._project_context())

        # 3. Always sprinkle community summaries — cheap and very useful for
        #    open-ended "describe Simone" style queries.
        chunks.extend(self._community_context(question))

        return chunks

    # ------------------------------------------------------------------

    def _project_context(self) -> List[RetrievedChunk]:
        rows = self._client.query(
            """
            MATCH (proj:PersonalProject)
            OPTIONAL MATCH (proj)-[r]->(t)
              WHERE t:Technology OR t:ProgrammingLanguage OR t:Topic OR t:Concept
            WITH proj, collect(DISTINCT coalesce(t.id, t.name)) AS related
            RETURN proj.name AS name, proj.url AS url, proj.description AS description,
                   proj.primary_language AS language, proj.stars AS stars,
                   proj.topics AS topics, related
            ORDER BY proj.stars DESC
            LIMIT 12
            """
        )
        chunks = []
        for row in rows:
            text = (
                f"Project: {row.get('name')}\n"
                f"URL: {row.get('url')}\n"
                f"Language: {row.get('language')} | Stars: {row.get('stars')}\n"
                f"Description: {row.get('description') or '(none)'}\n"
                f"Tech: {', '.join([t for t in (row.get('related') or []) if t])}"
            )
            chunks.append(RetrievedChunk(text=text, score=1.0, source="graph:project"))
        return chunks

    def _community_context(self, question: str, k: int = 2) -> List[RetrievedChunk]:
        try:
            embedding = get_embeddings().embed_query(question)
        except Exception:
            return []
        rows = self._client.query(
            """
            MATCH (c:CommunitySummary)
            WHERE c.embedding IS NOT NULL
            WITH c, gds.similarity.cosine(c.embedding, $emb) AS score
            ORDER BY score DESC LIMIT $k
            RETURN c.summary AS summary, score
            """,
            {"emb": embedding, "k": k},
        )
        if not rows:
            # Fallback: return top-N largest communities without similarity.
            rows = self._client.query(
                "MATCH (c:CommunitySummary) RETURN c.summary AS summary, c.size AS score "
                "ORDER BY c.size DESC LIMIT $k",
                {"k": k},
            )
        return [
            RetrievedChunk(text=r["summary"], score=float(r.get("score") or 0.0), source="graph:community")
            for r in rows
            if r.get("summary")
        ]


@lru_cache(maxsize=1)
def build_retriever() -> HybridGraphRetriever:
    """Process-wide cached retriever for the FastAPI app."""
    return HybridGraphRetriever()
