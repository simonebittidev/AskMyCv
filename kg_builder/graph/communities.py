"""GraphRAG-style community detection + summaries.

We try Neo4j GDS first (fast, server-side). If unavailable (e.g. Aura free
tier without GDS), we fall back to a Python clustering pass with networkx +
leidenalg. Each community gets an LLM-generated summary stored as a
``CommunitySummary`` node, enabling "global search" queries.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from kg_builder.extraction._llm import get_chat_llm, get_embeddings
from kg_builder.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


_SUMMARY_PROMPT = """\
You are summarising a community of related entities extracted from a knowledge \
graph about a single person's professional profile.

Given the entities and the relationships between them, write a concise (3-5 \
sentence) summary describing what this community represents (e.g. "Cloud and \
DevOps tooling", "Generative AI experience", "Education path"). Stay grounded \
in the data, no speculation."""


class CommunityBuilder:
    """Orchestrate detection + summary generation."""

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Return the number of CommunitySummary nodes written."""
        communities = self._detect_with_gds()
        if communities is None:
            logger.info("[communities] GDS unavailable, falling back to leidenalg")
            communities = self._detect_with_leidenalg()
        if not communities:
            logger.warning("[communities] no communities detected; skipping summaries")
            return 0

        return self._summarise(communities)

    # ------------------------------------------------------------------
    # Detection backends
    # ------------------------------------------------------------------

    def _detect_with_gds(self) -> Optional[Dict[int, List[str]]]:
        try:
            self._client.query(
                """
                CALL gds.graph.project.cypher(
                    'kg-leiden',
                    'MATCH (n) WHERE NOT n:Document AND NOT n:CommunitySummary RETURN id(n) AS id',
                    'MATCH (a)-[r]-(b) WHERE NOT a:Document AND NOT b:Document
                     RETURN id(a) AS source, id(b) AS target'
                )
                """
            )
        except Exception as exc:
            logger.info("[communities] gds.graph.project failed: %s", exc)
            return None

        try:
            rows = self._client.query(
                "CALL gds.leiden.write('kg-leiden', {writeProperty: 'community'}) "
                "YIELD communityCount RETURN communityCount"
            )
            logger.info("[communities] GDS Leiden produced %s communities", rows)
        except Exception as exc:
            logger.info("[communities] gds.leiden.write failed: %s", exc)
            self._drop_gds_graph()
            return None
        finally:
            self._drop_gds_graph()

        rows = self._client.query(
            """
            MATCH (n) WHERE n.community IS NOT NULL
            RETURN n.community AS cid, collect(coalesce(n.id, n.name, toString(id(n)))) AS members
            """
        )
        return {int(r["cid"]): r["members"] for r in rows}

    def _drop_gds_graph(self) -> None:
        try:
            self._client.query("CALL gds.graph.drop('kg-leiden', false) YIELD graphName RETURN graphName")
        except Exception:
            pass

    def _detect_with_leidenalg(self) -> Dict[int, List[str]]:
        try:
            import networkx as nx  # type: ignore
            import leidenalg  # type: ignore
            import igraph as ig  # type: ignore
        except ImportError:
            logger.warning("[communities] leidenalg/networkx not installed; skipping")
            return {}

        rows = self._client.query(
            """
            MATCH (a)-[r]-(b)
            WHERE NOT a:Document AND NOT b:Document
              AND NOT a:CommunitySummary AND NOT b:CommunitySummary
            RETURN coalesce(a.id, a.name, toString(id(a))) AS src,
                   coalesce(b.id, b.name, toString(id(b))) AS dst
            """
        )
        if not rows:
            return {}

        g = nx.Graph()
        for row in rows:
            g.add_edge(row["src"], row["dst"])

        ig_g = ig.Graph.from_networkx(g)
        partition = leidenalg.find_partition(ig_g, leidenalg.ModularityVertexPartition)
        node_names = ig_g.vs["_nx_name"]

        communities: Dict[int, List[str]] = {}
        for cid, cluster in enumerate(partition):
            communities[cid] = [node_names[i] for i in cluster]

        # Persist community ids to Neo4j for downstream querying.
        for cid, members in communities.items():
            self._client.query(
                """
                UNWIND $members AS member
                MATCH (n)
                WHERE coalesce(n.id, n.name, toString(id(n))) = member
                SET n.community = $cid
                """,
                {"members": members, "cid": cid},
            )
        return communities

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def _summarise(self, communities: Dict[int, List[str]]) -> int:
        llm = get_chat_llm(temperature=0.2)
        embeddings = get_embeddings()

        written = 0
        for cid, members in communities.items():
            if len(members) < 2:
                continue
            edges = self._fetch_edges(members)
            payload = (
                f"Community #{cid}\n"
                f"Members: {', '.join(members[:50])}\n"
                f"Relationships (sample):\n"
                + "\n".join(f"- {a} -[{rel}]-> {b}" for a, rel, b in edges[:80])
            )
            try:
                response = llm.invoke(
                    [SystemMessage(content=_SUMMARY_PROMPT), HumanMessage(content=payload)]
                )
                summary_text = response.content if hasattr(response, "content") else str(response)
            except Exception as exc:
                logger.warning("[communities] LLM failed for community %s: %s", cid, exc)
                continue

            try:
                vec = embeddings.embed_query(summary_text)
            except Exception:
                vec = None

            self._client.query(
                """
                MERGE (c:CommunitySummary {id: $cid})
                SET c.summary = $summary, c.embedding = $embedding, c.size = $size
                """,
                {
                    "cid": f"community-{cid}",
                    "summary": summary_text,
                    "embedding": vec,
                    "size": len(members),
                },
            )
            # Connect to a few representative members.
            self._client.query(
                """
                MATCH (c:CommunitySummary {id: $cid})
                UNWIND $members AS m
                MATCH (n) WHERE coalesce(n.id, n.name, toString(id(n))) = m
                MERGE (c)-[:SUMMARIZES]->(n)
                """,
                {"cid": f"community-{cid}", "members": members[:25]},
            )
            written += 1

        logger.info("[communities] wrote %d CommunitySummary nodes", written)
        return written

    def _fetch_edges(self, members: List[str]) -> List[Tuple[str, str, str]]:
        rows = self._client.query(
            """
            MATCH (a)-[r]->(b)
            WHERE coalesce(a.id, a.name, toString(id(a))) IN $members
              AND coalesce(b.id, b.name, toString(id(b))) IN $members
            RETURN coalesce(a.id, a.name, toString(id(a))) AS src,
                   type(r) AS rel,
                   coalesce(b.id, b.name, toString(id(b))) AS dst
            LIMIT 200
            """,
            {"members": members},
        )
        return [(r["src"], r["rel"], r["dst"]) for r in rows]


def build_community_summaries(client: Neo4jClient) -> int:
    return CommunityBuilder(client).run()
