"""Single point of contact with Neo4j for the kg_builder package."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator, List, Optional

from langchain_neo4j import Neo4jGraph

from kg_builder.config import Settings, get_settings

logger = logging.getLogger(__name__)


# Every entity label whose ``id`` should be unique in the graph. Adding a label
# here means ``MERGE (n:Label {id: ...})`` cannot produce duplicates.
_DEFAULT_CONSTRAINT_LABELS = (
    "Person",
    "PersonalProject",
    "Role",
    "Skill",
    "Technology",
    "ProgrammingLanguage",
    "Organization",
    "Language",
    "Concept",
    "Contact",
    "Certification",
    "Activity",
    "DateRange",
    "Proficiency",
    "Topic",
    "Location",
    "CommunitySummary",
)


class Neo4jClient:
    """Thin wrapper that owns a :class:`Neo4jGraph` and exposes helpers."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._graph: Optional[Neo4jGraph] = None

    @property
    def graph(self) -> Neo4jGraph:
        if self._graph is None:
            self._graph = Neo4jGraph(
                url=self._settings.neo4j_uri,
                username=self._settings.neo4j_username,
                password=self._settings.neo4j_password,
                enhanced_schema=True,
                refresh_schema=False,
                driver_config={"max_connection_lifetime": 180},
            )
        return self._graph

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query(self, cypher: str, params: Optional[dict] = None) -> List[dict]:
        return self.graph.query(cypher, params=params or {})

    def wipe(self) -> None:
        """Detach-delete every node — used at the start of a full rebuild."""
        logger.info("[neo4j] wiping graph")
        self.graph.query("MATCH (n) DETACH DELETE n")

    def ensure_constraints(self, labels: Optional[List[str]] = None) -> None:
        """Create uniqueness constraints on ``id`` for every entity label.

        With these constraints in place, ``MERGE (n:Label {id: ...})`` becomes
        atomic and no LLM-extracted duplicate can ever sneak in — even before
        canonicalisation runs.
        """
        labels = labels or list(_DEFAULT_CONSTRAINT_LABELS)
        for label in labels:
            constraint_name = f"{label.lower()}_id_unique"
            try:
                self.query(
                    f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("[neo4j] could not create constraint %s: %s", constraint_name, exc)
        logger.info("[neo4j] uniqueness constraints ensured for %d labels", len(labels))

    def refresh_schema(self) -> None:
        self.graph.refresh_schema()

    @contextmanager
    def session(self) -> Iterator[Any]:
        with self.graph._driver.session() as sess:
            yield sess

    def close(self) -> None:
        if self._graph is not None:
            try:
                self._graph._driver.close()
            except Exception:
                pass
            self._graph = None


@lru_cache(maxsize=1)
def get_default_client() -> Neo4jClient:
    """Process-wide cached client, suitable for re-use from FastAPI handlers."""
    return Neo4jClient()
