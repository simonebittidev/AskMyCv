"""End-to-end orchestration: ``run_pipeline()`` rebuilds the knowledge graph.

Pipeline stages:

1. **Wipe** the existing graph.
2. **Load PDFs** (CV + Cover Letter) → Markdown via Docling.
3. **Load GitHub READMEs** for the curated project list.
4. **Summarise + chunk + embed** each document (with late-chunking blend).
5. **Anchor PersonalProject** nodes deterministically (MERGE).
6. **LLM graph extraction** under per-source schemas.
7. **Persist** graph documents in Neo4j.
8. **Build community summaries** (Leiden) for GraphRAG-style global search.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import List, Sequence

from langchain_core.documents import Document

from kg_builder.config import Settings, get_settings
from kg_builder.extraction.chunker import chunk_document, embed_chunks
from kg_builder.extraction.graph_extractor import GraphExtractor
from kg_builder.extraction.summarizer import summarize_document
from kg_builder.graph.canonicalize import canonicalize_graph
from kg_builder.graph.communities import build_community_summaries
from kg_builder.graph.loader import GraphLoader
from kg_builder.graph.neo4j_client import Neo4jClient
from kg_builder.graph.schema import schema_for_kind
from kg_builder.models import ProjectDoc, SourceDocument
from kg_builder.sources import GitHubSource, PdfSource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-source processing
# ---------------------------------------------------------------------------


async def _process_source_document(
    source_doc: SourceDocument,
    extra_instructions: str = "",
) -> List[Document]:
    """Summarise → chunk → embed a single source document.

    Returns LangChain :class:`Document` chunks ready for graph extraction.
    """
    logger.info("[pipeline] processing %s (%s)", source_doc.source_id, source_doc.kind)
    summary = summarize_document(source_doc.content)
    chunks = chunk_document(summary.detailed_summary)
    return embed_chunks(
        full_markdown=source_doc.content,
        chunks=chunks,
        source_id=source_doc.source_id,
        kind=source_doc.kind,
    )


def _additional_context(brief_overview: str) -> str:
    today = date.today().isoformat()
    return (
        f"Document brief overview: {brief_overview}\n"
        f"Today's date: {today}\n"
        "Treat all chunks as extracts of the same logical document."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_pipeline(settings: Settings | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = settings or get_settings()

    client = Neo4jClient(settings)
    loader = GraphLoader(client, person_name=settings.person_name)

    # 1. Reset + uniqueness constraints. Constraints are DB-level and survive
    #    the wipe; recreating them here is idempotent and guarantees that any
    #    MERGE downstream cannot produce duplicate ids.
    client.wipe()
    client.ensure_constraints()
    loader.ensure_person()

    # 2. Sources.
    pdf_source = PdfSource(folder=settings.pdf_folder)
    github_source = GitHubSource(projects=settings.projects, token=settings.github_token)

    pdf_docs = await pdf_source.load()
    gh_docs = await github_source.load()
    logger.info("[pipeline] loaded %d PDFs and %d GitHub READMEs", len(pdf_docs), len(gh_docs))

    # 3. Anchor PersonalProject nodes BEFORE LLM extraction so the GitHub
    #    extractor can attach techs to existing project nodes.
    project_docs = _to_project_docs(gh_docs)
    if project_docs:
        loader.upsert_projects(project_docs)

    # 4. Process each document and extract graph documents per schema.
    await _ingest(pdf_docs, loader, person_name=settings.person_name)
    await _ingest(gh_docs, loader, person_name=settings.person_name)

    # 5. Canonicalise: merge Person variants ("Simone" + "Simone Bitti"),
    #    reclassify role-like Person nodes as Role, dedupe Skill/Technology
    #    by case-insensitive id, drop orphans.
    try:
        canonicalize_graph(client, settings.person_name)
    except Exception as exc:
        logger.warning("[pipeline] canonicalisation failed: %s", exc)

    # 6. Communities / GraphRAG global search.
    if settings.enable_communities:
        try:
            n = build_community_summaries(client)
            logger.info("[pipeline] %d community summaries written", n)
        except Exception as exc:
            logger.warning("[pipeline] community detection failed: %s", exc)

    client.refresh_schema()
    client.close()
    logger.info("[pipeline] done")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _ingest(
    source_docs: Sequence[SourceDocument],
    loader: GraphLoader,
    *,
    person_name: str,
) -> None:
    for source_doc in source_docs:
        try:
            chunks = await _process_source_document(source_doc)
        except Exception as exc:
            logger.exception("[pipeline] processing failed for %s: %s", source_doc.source_id, exc)
            continue

        schema = schema_for_kind(source_doc.kind)
        extra = _build_extra_instructions(source_doc, person_name)
        extractor = GraphExtractor(schema, additional_instructions=extra)
        try:
            result = await extractor.extract(chunks)
        except Exception as exc:
            logger.exception("[pipeline] extraction failed for %s: %s", source_doc.source_id, exc)
            continue

        loader.load_graph_documents(result.graph_documents)


def _build_extra_instructions(source_doc: SourceDocument, person_name: str) -> str:
    """Strong, per-source instructions to keep the extracted graph clean.

    Targets the dedupe issues we observed in practice: multiple Person nodes
    for the same person, and Person nodes for what are actually job titles.
    """
    if source_doc.kind == "github_readme":
        return (
            f"This README belongs to the project '{source_doc.source_id}' "
            f"(node id '{source_doc.source_id}'). When you extract Technology, "
            "ProgrammingLanguage, Concept or Topic entities, relate them to that "
            "project. Do NOT create a Person node — the project owner is "
            "managed separately. Do NOT invent additional PersonalProject nodes."
        )

    # CV / cover letter
    return (
        f"This document describes EXACTLY ONE person: '{person_name}'.\n"
        f"- Always use the EXACT id '{person_name}' for the Person node — "
        "never abbreviate, never split, never use just the first name.\n"
        "- There MUST be at most one Person node in your output, with id "
        f"'{person_name}'.\n"
        "- Job titles like 'Software Engineer', 'AI Engineer', 'Consultant' are "
        "NEVER Person nodes — they are Role nodes connected to the Person via "
        "WORKED_AS.\n"
        "- Companies, universities and clients are Organization nodes, not "
        "Person nodes.\n"
        "- Use canonical Title Case for all entity ids (e.g. 'Python', not "
        "'python' or 'PYTHON'; 'Microsoft Azure', not 'azure').\n"
        "- Do NOT extract PersonalProject, Project or ProjectUrl nodes — "
        "personal projects are ingested from a separate GitHub source."
    )


def _to_project_docs(source_docs: Sequence[SourceDocument]) -> List[ProjectDoc]:
    projects: List[ProjectDoc] = []
    for source_doc in source_docs:
        if source_doc.kind != "github_readme":
            continue
        try:
            projects.append(ProjectDoc.model_validate(source_doc.metadata))
        except Exception as exc:
            logger.warning("[pipeline] could not normalise project %s: %s", source_doc.source_id, exc)
    return projects
