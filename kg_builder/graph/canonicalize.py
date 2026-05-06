"""Post-ingest canonicalisation of the knowledge graph.

The LLM extractor frequently produces:

* Multiple ``Person`` nodes for the same person (e.g. ``Simone`` vs.
  ``Simone Bitti``).
* ``Person`` nodes that are actually job titles (e.g. ``Software Engineer``).
* Casing/whitespace duplicates of ``Skill``/``Technology`` entities
  (``Python`` vs. ``python`` vs. ``PYTHON``).

This module runs a deterministic clean-up pass after extraction to merge those
duplicates and reclassify mislabelled nodes. Pure Cypher — no APOC required —
so it works on Neo4j Aura free tier.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List

from kg_builder.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


# Heuristic patterns for "this Person node is actually a Role".
_ROLE_PATTERNS = re.compile(
    r"\b(engineer|developer|architect|consultant|analyst|scientist|manager|"
    r"lead|specialist|administrator|designer|intern|tester|qa|founder|"
    r"co-?founder|ceo|cto|cio|coo|director|head|owner|programmer|technician|"
    r"researcher)\b",
    re.IGNORECASE,
)

# Labels we deduplicate by case-insensitive id.
_DEDUPE_LABELS = ("Skill", "Technology", "ProgrammingLanguage", "Concept", "Topic", "Organization", "Role")


def canonicalize_graph(client: Neo4jClient, person_name: str) -> None:
    """Run the full canonicalisation pass."""
    logger.info("[canonicalize] starting clean-up for person='%s'", person_name)
    _ensure_canonical_person(client, person_name)
    _reclassify_role_persons(client)
    _merge_person_variants(client, person_name)
    for label in _DEDUPE_LABELS:
        _merge_case_insensitive_duplicates(client, label)
    _drop_orphans(client)
    logger.info("[canonicalize] done")


# ---------------------------------------------------------------------------
# Canonical Person
# ---------------------------------------------------------------------------


def _ensure_canonical_person(client: Neo4jClient, person_name: str) -> None:
    client.query(
        "MERGE (p:Person {id: $name}) "
        "ON CREATE SET p.name = $name "
        "ON MATCH SET p.name = coalesce(p.name, $name)",
        {"name": person_name},
    )


def _person_variants(person_name: str) -> List[str]:
    """Variants the LLM is likely to use for the same person."""
    parts = person_name.split()
    variants = {person_name, person_name.lower(), person_name.upper(), person_name.title()}
    if parts:
        variants.add(parts[0])  # first name
        variants.add(parts[0].lower())
        variants.add(parts[0].title())
    return [v for v in variants if v]


def _merge_person_variants(client: Neo4jClient, person_name: str) -> None:
    variants = _person_variants(person_name)
    rows = client.query(
        """
        MATCH (other:Person)
        WHERE other.id <> $canonical
          AND (toLower(other.id) IN $variants OR toLower(coalesce(other.name, '')) IN $variants)
        RETURN other.id AS id
        """,
        {"canonical": person_name, "variants": [v.lower() for v in variants]},
    )
    duplicates = [r["id"] for r in rows if r["id"]]
    if not duplicates:
        return
    logger.info("[canonicalize] merging %d Person duplicates into '%s': %s",
                len(duplicates), person_name, duplicates)
    for dup_id in duplicates:
        _merge_node_into(client, label="Person", source_id=dup_id, target_id=person_name)


# ---------------------------------------------------------------------------
# Reclassify Person → Role when the id looks like a job title
# ---------------------------------------------------------------------------


def _reclassify_role_persons(client: Neo4jClient) -> None:
    rows = client.query(
        "MATCH (p:Person) RETURN p.id AS id, coalesce(p.name, p.id) AS name"
    )
    targets = [r for r in rows if r.get("id") and _looks_like_role(r["name"] or r["id"])]
    if not targets:
        return

    for row in targets:
        node_id = row["id"]
        logger.info("[canonicalize] reclassifying Person->Role: %s", node_id)
        # Add :Role label, drop :Person label, dedupe with existing Role of same id.
        client.query(
            """
            MATCH (p:Person {id: $id})
            REMOVE p:Person
            SET p:Role
            """,
            {"id": node_id},
        )
        # If a canonical Role with the same id already exists, fold them together.
        existing = client.query(
            "MATCH (r:Role) WHERE r.id = $id RETURN count(r) AS n", {"id": node_id}
        )
        if existing and existing[0]["n"] > 1:
            _merge_case_insensitive_duplicates(client, "Role")


def _looks_like_role(text: str) -> bool:
    if not text:
        return False
    if _ROLE_PATTERNS.search(text):
        return True
    # Multi-word titles are usually roles, single-word names usually aren't.
    return False


# ---------------------------------------------------------------------------
# Generic duplicate merge by case-insensitive id
# ---------------------------------------------------------------------------


def _merge_case_insensitive_duplicates(client: Neo4jClient, label: str) -> None:
    groups = client.query(
        f"""
        MATCH (n:{label}) WHERE n.id IS NOT NULL
        WITH toLower(trim(n.id)) AS key, collect(n.id) AS ids
        WHERE size(ids) > 1
        RETURN key, ids
        """
    )
    for group in groups:
        ids: List[str] = group["ids"]
        # Pick the prettiest id as canonical: prefer Title Case, then longest.
        canonical = sorted(ids, key=lambda s: (-_canonical_score(s), -len(s)))[0]
        for other in ids:
            if other == canonical:
                continue
            _merge_node_into(client, label=label, source_id=other, target_id=canonical)
        logger.info("[canonicalize] %s: merged %s -> %s", label, ids, canonical)


def _canonical_score(s: str) -> int:
    """Higher is better. Prefer mixed-case (likely Title Case) over all-lower/all-upper."""
    if s != s.lower() and s != s.upper():
        return 2
    if s == s.title():
        return 1
    return 0


# ---------------------------------------------------------------------------
# Pure-Cypher node merge (no APOC dependency)
# ---------------------------------------------------------------------------


def _merge_node_into(client: Neo4jClient, *, label: str, source_id: str, target_id: str) -> None:
    """Move every relationship from ``source`` into ``target`` and delete ``source``.

    Uses dynamic Cypher generated per relationship type — works without APOC.
    """
    if source_id == target_id:
        return

    # Discover relationship types attached to the source node.
    rel_rows = client.query(
        f"""
        MATCH (s:{label} {{id: $sid}})-[r]-(other)
        RETURN type(r) AS rel_type,
               CASE WHEN startNode(r) = s THEN 'out' ELSE 'in' END AS direction
        """,
        {"sid": source_id},
    )
    seen: set[tuple[str, str]] = set()
    rels: List[tuple[str, str]] = []
    for row in rel_rows:
        key = (row["rel_type"], row["direction"])
        if key not in seen:
            seen.add(key)
            rels.append(key)

    for rel_type, direction in rels:
        if direction == "out":
            cypher = (
                f"MATCH (s:{label} {{id: $sid}})-[r:`{rel_type}`]->(other) "
                f"MATCH (t:{label} {{id: $tid}}) "
                f"MERGE (t)-[:`{rel_type}`]->(other) "
                "DELETE r"
            )
        else:
            cypher = (
                f"MATCH (other)-[r:`{rel_type}`]->(s:{label} {{id: $sid}}) "
                f"MATCH (t:{label} {{id: $tid}}) "
                f"MERGE (other)-[:`{rel_type}`]->(t) "
                "DELETE r"
            )
        client.query(cypher, {"sid": source_id, "tid": target_id})

    # Source node has no more useful relationships; LLM-extracted nodes
    # typically only carry id/name properties so we drop them as-is.
    client.query(
        f"MATCH (s:{label} {{id: $sid}}) DETACH DELETE s",
        {"sid": source_id},
    )


# ---------------------------------------------------------------------------
# Orphan cleanup
# ---------------------------------------------------------------------------


def _drop_orphans(client: Neo4jClient) -> None:
    """Remove nodes that became disconnected after merging (excluding chunks)."""
    client.query(
        """
        MATCH (n)
        WHERE NOT (n)--()
          AND NOT n:Document
          AND NOT n:CommunitySummary
          AND NOT n:Person
        DELETE n
        """
    )
