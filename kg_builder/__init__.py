"""Knowledge Graph builder package for ChatMyCv.

Modular pipeline that ingests heterogeneous sources (PDFs, GitHub README files)
and loads a structured + vectorized knowledge graph into Neo4j.
"""

from kg_builder.pipeline import run_pipeline

__all__ = ["run_pipeline"]
