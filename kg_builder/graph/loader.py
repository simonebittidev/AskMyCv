"""Load graph documents into Neo4j with deterministic project anchoring."""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from kg_builder.graph.neo4j_client import Neo4jClient
from kg_builder.models import ProjectDoc

logger = logging.getLogger(__name__)


class GraphLoader:
    """Persist graph documents and anchor :class:`PersonalProject` nodes."""

    def __init__(self, client: Neo4jClient, person_name: str) -> None:
        self._client = client
        self._person_name = person_name

    # ------------------------------------------------------------------
    # Person anchor
    # ------------------------------------------------------------------

    def ensure_person(self) -> None:
        self._client.query(
            "MERGE (p:Person {id: $name}) "
            "ON CREATE SET p.name = $name "
            "ON MATCH SET p.name = $name",
            {"name": self._person_name},
        )

    # ------------------------------------------------------------------
    # PersonalProject deterministic creation
    # ------------------------------------------------------------------

    def upsert_projects(self, projects: Iterable[ProjectDoc]) -> None:
        """Idempotently MERGE :class:`PersonalProject` nodes from GitHub metadata."""
        for project in projects:
            self._client.query(
                """
                MERGE (proj:PersonalProject {id: $repo})
                SET proj.name = $name,
                    proj.display_name = $display_name,
                    proj.url = $url,
                    proj.description = $description,
                    proj.homepage = $homepage,
                    proj.primary_language = $primary_language,
                    proj.stars = $stars,
                    proj.last_push = $pushed_at,
                    proj.topics = $topics
                WITH proj
                MATCH (person:Person {id: $person})
                MERGE (person)-[:CREATED]->(proj)
                """,
                {
                    "repo": project.repo,
                    "name": project.name,
                    "display_name": project.display_name,
                    "url": project.url,
                    "description": project.description,
                    "homepage": project.homepage,
                    "primary_language": project.primary_language,
                    "stars": project.stars,
                    "pushed_at": project.pushed_at.isoformat() if project.pushed_at else None,
                    "topics": project.topics,
                    "person": self._person_name,
                },
            )

            # Topics get their own nodes for richer retrieval.
            for topic in project.topics or []:
                self._client.query(
                    """
                    MERGE (t:Topic {id: $topic})
                    WITH t
                    MATCH (proj:PersonalProject {id: $repo})
                    MERGE (proj)-[:HAS_TOPIC]->(t)
                    """,
                    {"topic": topic, "repo": project.repo},
                )
        logger.info("[loader] upserted %d projects", len(list(projects)) if hasattr(projects, "__len__") else -1)

    # ------------------------------------------------------------------
    # Generic graph documents
    # ------------------------------------------------------------------

    def load_graph_documents(self, graph_documents: list, *, include_source: bool = True) -> None:
        if not graph_documents:
            return
        self._client.graph.add_graph_documents(
            graph_documents,
            include_source=include_source,
            baseEntityLabel=False,
        )
        logger.info("[loader] persisted %d graph documents", len(graph_documents))

    # ------------------------------------------------------------------
    # Connect README extractions to the anchored PersonalProject
    # ------------------------------------------------------------------

    def link_project_extractions(
        self,
        repo: str,
        techs: Optional[List[str]] = None,
        relationship: str = "USES",
    ) -> None:
        """Optional helper: explicitly relate extracted Technology nodes to a project."""
        for tech in techs or []:
            self._client.query(
                f"""
                MATCH (proj:PersonalProject {{id: $repo}})
                MERGE (t:Technology {{id: $tech}})
                MERGE (proj)-[:{relationship}]->(t)
                """,
                {"repo": repo, "tech": tech},
            )
