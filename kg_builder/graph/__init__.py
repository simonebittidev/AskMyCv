"""Neo4j graph access + loaders + community detection."""

from kg_builder.graph.canonicalize import canonicalize_graph
from kg_builder.graph.communities import build_community_summaries
from kg_builder.graph.loader import GraphLoader
from kg_builder.graph.neo4j_client import Neo4jClient
from kg_builder.graph.schema import (
    CV_SCHEMA,
    GITHUB_SCHEMA,
    SourceSchema,
    schema_for_kind,
)

__all__ = [
    "Neo4jClient",
    "GraphLoader",
    "SourceSchema",
    "CV_SCHEMA",
    "GITHUB_SCHEMA",
    "schema_for_kind",
    "build_community_summaries",
    "canonicalize_graph",
]
