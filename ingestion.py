"""
Ingestion pipeline:
  PDF -> Markdown (marker-pdf)
  -> custom header/paragraph aware chunking with breadcrumb
  -> contextual chunking (LLM situates each chunk in the full doc)
  -> STRICT entity/relation extraction with neo4j-graphrag (v1.15+)
  -> Neo4j persistence with neo4j-graphrag's writer

Run:
    python ingestion.py
"""

import asyncio
import os
import re
from datetime import date
from pathlib import Path
from typing import List, Optional

import base64

import fitz  # pymupdf
import neo4j
import openai
from dotenv import load_dotenv
from pydantic import BaseModel

from neo4j_graphrag.embeddings.openai import AzureOpenAIEmbeddings as _AzureOpenAIEmbeddings

from entity_resolver import AzureEmbeddingResolver


class AzureOpenAIEmbeddings(_AzureOpenAIEmbeddings):
    """Wrapper that bakes `dimensions` into every embed_query call.

    The base class forwards constructor kwargs to the underlying AzureOpenAI
    client, which does NOT accept `dimensions` — that parameter belongs to the
    `embeddings.create()` request.
    """

    def __init__(self, *args, dimensions: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._dimensions = dimensions

    def embed_query(self, text: str, **kwargs):
        if self._dimensions is not None and "dimensions" not in kwargs:
            kwargs["dimensions"] = self._dimensions
        return super().embed_query(text, **kwargs)
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.resolver import (
    SinglePropertyExactMatchResolver,
)
from neo4j_graphrag.experimental.components.schema import (
    NodeType as SchemaNodeType,
    PropertyType,
    RelationshipType as SchemaRelationshipType,
    SchemaBuilder,
)
from neo4j_graphrag.experimental.components.types import (
    DocumentInfo,
    LexicalGraphConfig,
    TextChunk,
    TextChunks,
)

load_dotenv()




FILES_FOLDER = "files/Simone Bitti"

# ─────────────────────────────────────────────────────────────────────────────
# Strict schema definition
# ─────────────────────────────────────────────────────────────────────────────

NODE_TYPES = [
    # — People & organisations
    SchemaNodeType(label="Person",              properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Role",                properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Organization",        properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Location",            properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="country", type="STRING"), PropertyType(name="description", type="STRING")]),
    # — IT-specific technical nodes
    SchemaNodeType(label="ProgrammingLanguage", properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Framework",           properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Database",            properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="CloudPlatform",       properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="DevOpsTool",          properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Technology",          properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Domain",              properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Methodology",         properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    # — Skills & languages
    SchemaNodeType(label="Skill",               properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Language",            properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Proficiency",         properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    # — Education & credentials
    SchemaNodeType(label="Degree",              properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Certification",       properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Award",               properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    # — Projects & activities
    SchemaNodeType(label="Project",             properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="PersonalProject",     properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="ProjectUrl",          properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Activity",            properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    # — Contact & time
    SchemaNodeType(label="Contact",             properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="type", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="DateRange",           properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
]

_TODAY = date.today().strftime("%Y-%m-%d")

RELATIONSHIP_TYPES = [
    SchemaRelationshipType(
        label="WORKS_AT",
        description=(
            f"CURRENT employment (Person→Organization). Use ONLY when the position is ongoing as of today ({_TODAY}): "
            "the chunk says 'Present', 'Current', 'Now', 'ongoing', or has a start date with NO end date. "
            "If unsure, prefer WORKED_AT."
        ),
        properties=[PropertyType(name="description", type="STRING")],
    ),
    SchemaRelationshipType(
        label="WORKED_AT",
        description=(
            f"PAST employment (Person→Organization). Use when the end date is explicitly BEFORE today ({_TODAY}). "
            "Default choice when there is any doubt about whether the role is still active."
        ),
        properties=[PropertyType(name="description", type="STRING")],
    ),
    SchemaRelationshipType(label="HAS_ROLE",        description="Links a Person to a Role they hold or held (e.g. 'Software Engineer', 'Data Scientist').",        properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="ROLE_AT",         description="Links a Role to the Organization where it was or is held.",                                         properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HAS_SKILL",       description="Links a Person to a Skill (technical or soft skill).",                                              properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HAS_PROFICIENCY", description="Links a Skill or Language node to a Proficiency level (e.g. Beginner, Intermediate, Advanced, Fluent, Native).", properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="KNOWS_LANGUAGE",  description="A Person knows a natural Language (Italian, English, French, ...).",                                properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HAS_CERTIFICATION", description="A Person holds a Certification.",                                                                 properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HAS_AWARD",       description="A Person received an Award or recognition.",                                                        properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="STUDIED_AT",      description="A Person studied at an Organization (university, school, bootcamp).",                               properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="OBTAINED_DEGREE", description="A Person obtained a Degree (e.g. 'Bachelor of Science in Computer Science').",                      properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="GRANTED_BY",      description="A Certification, Award, or Degree was issued/granted by an Organization.",                          properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HAS_CONTACT",     description="A Person has a Contact (email, phone, LinkedIn, GitHub, website). Set the `type` property on the Contact node.", properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="INVOLVED_IN",     description="A Person was involved in a Project or PersonalProject.",                                            properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="PARTICIPATED_IN", description="A Person participated in an Activity (volunteering, sport, association, event).",                   properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="LOCATED_IN",      description="A Person or Organization is located in a Location (city, country, region).",                        properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="USES",            description="A Role, Project, or PersonalProject uses a Technology, Framework, Database, or DevOpsTool. Use more specific relations (DEPLOYED_ON, APPLIES) when applicable.", properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="DEPLOYED_ON",     description="A Role, Project, or PersonalProject is deployed on or hosted by a CloudPlatform (e.g. AWS, Azure, GCP).", properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="SPECIALIZES_IN",  description="A Person or Role specializes in a Domain (e.g. Backend, Machine Learning, Data Engineering, NLP, DevOps, Frontend).", properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="APPLIES",         description="A Role or Project applies a Methodology or practice (e.g. Agile, Scrum, TDD, REST, Microservices, CI/CD).", properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="DURING",          description="Links a Role, Project, PersonalProject, Certification, Degree, Award, or Activity to its DateRange (e.g. '2020–2023', 'Jan 2021 – Present').", properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HAS_URL",         description="Links a Project or PersonalProject to a ProjectUrl.",                                               properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="PART_OF",         description="Links a Project to the Organization it was developed for or within.",                               properties=[PropertyType(name="description", type="STRING")]),
]

# (source_label, relationship_label, target_label)
PATTERNS = [
    # Person — employment
    ("Person",          "WORKS_AT",         "Organization"),
    ("Person",          "WORKED_AT",        "Organization"),
    ("Person",          "HAS_ROLE",         "Role"),
    # Role
    ("Role",            "ROLE_AT",          "Organization"),
    ("Role",            "WORKS_AT",         "Organization"),
    ("Role",            "SPECIALIZES_IN",   "Domain"),
    ("Role",            "USES",             "ProgrammingLanguage"),
    ("Role",            "USES",             "Framework"),
    ("Role",            "USES",             "Database"),
    ("Role",            "USES",             "DevOpsTool"),
    ("Role",            "USES",             "Technology"),
    ("Role",            "DEPLOYED_ON",      "CloudPlatform"),
    ("Role",            "APPLIES",          "Methodology"),
    ("Role",            "DURING",           "DateRange"),
    # Person — IT specialisation
    ("Person",          "SPECIALIZES_IN",   "Domain"),
    # Person — skills & languages
    ("Person",          "HAS_SKILL",        "Skill"),
    ("Person",          "KNOWS_LANGUAGE",   "Language"),
    ("Skill",           "HAS_PROFICIENCY",  "Proficiency"),
    ("Language",        "HAS_PROFICIENCY",  "Proficiency"),
    # Person — education
    ("Person",          "STUDIED_AT",       "Organization"),
    ("Person",          "OBTAINED_DEGREE",  "Degree"),
    ("Degree",          "GRANTED_BY",       "Organization"),
    ("Degree",          "DURING",           "DateRange"),
    # Person — certifications
    ("Person",          "HAS_CERTIFICATION","Certification"),
    ("Certification",   "GRANTED_BY",       "Organization"),
    ("Certification",   "DURING",           "DateRange"),
    # Person — awards
    ("Person",          "HAS_AWARD",        "Award"),
    ("Award",           "GRANTED_BY",       "Organization"),
    ("Award",           "DURING",           "DateRange"),
    # Person — contacts & location
    ("Person",          "HAS_CONTACT",      "Contact"),
    ("Person",          "LOCATED_IN",       "Location"),
    ("Organization",    "LOCATED_IN",       "Location"),
    # Person — activities
    ("Person",          "PARTICIPATED_IN",  "Activity"),
    ("Activity",        "DURING",           "DateRange"),
    # Projects
    ("Person",          "INVOLVED_IN",      "Project"),
    ("Person",          "INVOLVED_IN",      "PersonalProject"),
    ("Project",         "USES",             "ProgrammingLanguage"),
    ("Project",         "USES",             "Framework"),
    ("Project",         "USES",             "Database"),
    ("Project",         "USES",             "DevOpsTool"),
    ("Project",         "USES",             "Technology"),
    ("Project",         "DEPLOYED_ON",      "CloudPlatform"),
    ("Project",         "APPLIES",          "Methodology"),
    ("Project",         "DURING",           "DateRange"),
    ("Project",         "HAS_URL",          "ProjectUrl"),
    ("Project",         "PART_OF",          "Organization"),
    ("PersonalProject", "USES",             "ProgrammingLanguage"),
    ("PersonalProject", "USES",             "Framework"),
    ("PersonalProject", "USES",             "Database"),
    ("PersonalProject", "USES",             "DevOpsTool"),
    ("PersonalProject", "USES",             "Technology"),
    ("PersonalProject", "DEPLOYED_ON",      "CloudPlatform"),
    ("PersonalProject", "APPLIES",          "Methodology"),
    ("PersonalProject", "DURING",           "DateRange"),
    ("PersonalProject", "HAS_URL",          "ProjectUrl"),
]


# ─────────────────────────────────────────────────────────────────────────────
# GitHub README schema — nodes, relationships and allowed patterns
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_NODE_TYPES = [
    SchemaNodeType(label="PersonalProject",    properties=[PropertyType(name="name", type="STRING"), PropertyType(name="url", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Technology",         properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="ProgrammingLanguage",properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Framework",          properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Database",           properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="CloudPlatform",      properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="DevOpsTool",         properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Concept",            properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Topic",              properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Organization",       properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
]

GITHUB_RELATIONSHIP_TYPES = [
    SchemaRelationshipType(label="USES",            description="PersonalProject uses a Technology or Framework.",                                    properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="BUILT_WITH",      description="PersonalProject is built with a Technology, Framework, or ProgrammingLanguage.",    properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="WRITTEN_IN",      description="PersonalProject is primarily written in a ProgrammingLanguage.",                    properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="DEPENDS_ON",      description="PersonalProject depends on a Technology or Framework.",                             properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="INTEGRATES_WITH", description="PersonalProject integrates with an external Technology, API, or Organization.",     properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="DEPLOYED_ON",     description="PersonalProject is deployed on or hosted by a CloudPlatform.",                      properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HOSTED_ON",       description="PersonalProject is hosted on a Technology or Organization platform.",               properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="IMPLEMENTS",      description="PersonalProject implements a Concept, pattern, or algorithm.",                      properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="SOLVES",          description="PersonalProject solves a problem described by a Concept.",                          properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="DEMONSTRATES",    description="PersonalProject demonstrates a Concept or Skill.",                                  properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HAS_TOPIC",       description="PersonalProject is tagged with a Topic.",                                           properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="RELATED_TO",      description="A Technology or Concept is semantically related to another.",                       properties=[PropertyType(name="description", type="STRING")]),
]

GITHUB_PATTERNS = [
    ("PersonalProject", "USES",            "Technology"),
    ("PersonalProject", "USES",            "Framework"),
    ("PersonalProject", "USES",            "Database"),
    ("PersonalProject", "USES",            "DevOpsTool"),
    ("PersonalProject", "BUILT_WITH",      "Technology"),
    ("PersonalProject", "BUILT_WITH",      "Framework"),
    ("PersonalProject", "BUILT_WITH",      "ProgrammingLanguage"),
    ("PersonalProject", "WRITTEN_IN",      "ProgrammingLanguage"),
    ("PersonalProject", "DEPENDS_ON",      "Technology"),
    ("PersonalProject", "DEPENDS_ON",      "Framework"),
    ("PersonalProject", "INTEGRATES_WITH", "Technology"),
    ("PersonalProject", "INTEGRATES_WITH", "Organization"),
    ("PersonalProject", "DEPLOYED_ON",     "CloudPlatform"),
    ("PersonalProject", "HOSTED_ON",       "Technology"),
    ("PersonalProject", "HOSTED_ON",       "Organization"),
    ("PersonalProject", "IMPLEMENTS",      "Concept"),
    ("PersonalProject", "SOLVES",          "Concept"),
    ("PersonalProject", "DEMONSTRATES",    "Concept"),
    ("PersonalProject", "HAS_TOPIC",       "Topic"),
    ("Technology",      "RELATED_TO",      "Technology"),
    ("Concept",         "RELATED_TO",      "Concept"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — PDF to Markdown
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_markdown(pdf_path: str) -> str:
    client = openai.AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2025-01-01-preview",
    )
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        img_bytes = page.get_pixmap(dpi=200).tobytes("png")
        b64 = base64.b64encode(img_bytes).decode()
        resp = client.chat.completions.create(
            model="gpt-5.4",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": (
                        "Estrai tutto il testo da questa pagina di CV in ordine logico "
                        "(prima la colonna sinistra dall'alto in basso, poi la colonna destra). "
                        "Restituisci solo il testo estratto in formato Markdown, senza commenti aggiuntivi."
                    )},
                ],
            }],
            max_completion_tokens=4096,
        )
        parts.append(resp.choices[0].message.content or "")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Markdown chunking (header-aware, paragraph fallback)
# ─────────────────────────────────────────────────────────────────────────────

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class MarkdownSection(BaseModel):
    breadcrumb: str
    content: str
    order: int


_HEADERS_TO_SPLIT = [
    ("#", "h1"), ("##", "h2"), ("###", "h3"),
    ("####", "h4"), ("#####", "h5"), ("######", "h6"),
]


def split_markdown(md: str, chunk_size: int = 1800, chunk_overlap: int = 100) -> List[MarkdownSection]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = char_splitter.split_documents(header_splitter.split_text(md))
    out = []
    for order, doc in enumerate(docs):
        headers = [doc.metadata[k] for k in ["h1", "h2", "h3", "h4", "h5", "h6"] if doc.metadata.get(k)]
        breadcrumb = " > ".join(headers) if headers else "(root)"
        content = doc.page_content.strip()
        if content:
            out.append(MarkdownSection(breadcrumb=breadcrumb, content=content, order=order))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Contextual chunking (Anthropic-style)
# ─────────────────────────────────────────────────────────────────────────────

CONTEXT_PROMPT = """You are an expert at situating excerpts within their source document.

You will be given the full document and one of its chunks. Write 2-3 short sentences that situate the chunk inside the document, so a reader (or a retrieval system) can understand what this chunk is about and how it fits in the bigger picture.

Rules:
- Output ONLY the contextualization sentences. No preamble, no quotes, no bullets.
- Do not summarize the whole document — only what's needed to place this chunk.
- Use the same language as the document.
- Keep it under 60 words."""

# Separate prompt for GitHub READMEs: always English, ties the chunk to the CV person.
GITHUB_CONTEXT_PROMPT = """You are an expert at situating excerpts within technical project documentation.

You will be given the full README of a personal GitHub project and one of its chunks. Write 2-3 short sentences IN ENGLISH that situate the chunk within the project, making clear this is a personal project by {person_name}.

Rules:
- Output ONLY the contextualization sentences. No preamble, no quotes, no bullets.
- ALWAYS write in ENGLISH, regardless of the language of the source document.
- Do not summarize the whole document — only what's needed to place this chunk.
- Keep it under 60 words."""


PERSON_NAME_PROMPT = """You are given the full text of a CV or Cover Letter. Identify the person the document refers to (the candidate / author).

Rules:
- Output ONLY the person's full name, nothing else.
- No quotes, no preamble, no labels.
- If multiple names appear, pick the one the document is ABOUT (the candidate), not third parties (references, managers, etc.).
- If genuinely impossible to determine, output exactly: UNKNOWN"""


async def extract_person_name(llm: AzureOpenAILLM, full_doc: str) -> str:
    response = await llm.ainvoke(input=full_doc, system_instruction=PERSON_NAME_PROMPT)
    name = response.content.strip().strip('"').strip("'")
    return name or "UNKNOWN"


async def contextualize(llm: AzureOpenAILLM, full_doc: str, section: MarkdownSection) -> str:
    user = (
        f"<document>\n{full_doc}\n</document>\n\n"
        f"<chunk>\nSection path: {section.breadcrumb}\n\n{section.content}\n</chunk>"
    )
    response = await llm.ainvoke(input=user, system_instruction=CONTEXT_PROMPT)
    return response.content.strip()


async def contextualize_github(
    llm: AzureOpenAILLM, full_doc: str, section: MarkdownSection, person_name: str
) -> str:
    user = (
        f"<document>\n{full_doc}\n</document>\n\n"
        f"<chunk>\nSection path: {section.breadcrumb}\n\n{section.content}\n</chunk>"
    )
    prompt = GITHUB_CONTEXT_PROMPT.format(person_name=person_name)
    response = await llm.ainvoke(input=user, system_instruction=prompt)
    return response.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# GitHub ingestion
# ─────────────────────────────────────────────────────────────────────────────


async def run_github_ingestion(
    driver: neo4j.Driver,
    llm: AzureOpenAILLM,
    embedder,
    person_name: str = "Simone Bitti",
    github_token: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> None:
    """Fetch GitHub READMEs and ingest them into the knowledge graph.

    For each repo listed in ``config/projects.yml``:
    1. Anchor a PersonalProject node via MERGE and link it to the Person node.
    2. Chunk and contextualise the README in English, referencing the CV person.
    3. Extract entities/relations under the GitHub-specific strict schema.
       Person nodes are explicitly forbidden to avoid polluting the CV person.
    4. Write the resulting graph documents to Neo4j.
    """
    from github_source import fetch_github_projects, load_project_refs

    config = config_path or Path(__file__).parent / "config" / "projects.yml"
    refs = load_project_refs(config)
    if not refs:
        print("[github] no projects configured — check config/projects.yml")
        return

    projects = await fetch_github_projects(refs, token=github_token)
    if not projects:
        print("[github] no projects fetched")
        return

    schema_builder = SchemaBuilder()
    schema = await schema_builder.run(
        node_types=GITHUB_NODE_TYPES,
        relationship_types=GITHUB_RELATIONSHIP_TYPES,
        patterns=GITHUB_PATTERNS,
        additional_node_types=False,
        additional_relationship_types=False,
        additional_patterns=False,
    )

    text_chunk_embedder = TextChunkEmbedder(embedder=embedder)
    lexical_graph_config = LexicalGraphConfig()
    extractor = LLMEntityRelationExtractor(
        llm=llm,
        on_error=OnError.IGNORE,
        create_lexical_graph=True,
        use_structured_output=False,
    )
    writer = Neo4jWriter(driver=driver)
    today = date.today().strftime("%Y-%m-%d")

    for project in projects:
        label = project.display_name or project.name
        print(f"\n=== GitHub: {label} ({project.repo}) ===")

        md = project.to_markdown()
        sections = split_markdown(md)
        print(f"  README sections: {len(sections)}")

        # The provenance line ties each chunk to the CV person and explicitly
        # forbids the LLM from creating Person nodes during GitHub extraction.
        provenance_line = (
            f"This chunk comes from the GitHub README of \"{label}\" "
            f"(repository: {project.repo}), a personal project created by {person_name}. "
            f"IMPORTANT: do NOT create Person nodes — {person_name} is managed "
            f"separately by the CV pipeline."
        )

        text_chunks: list[TextChunk] = []
        for sec in sections:
            ctx = await contextualize_github(llm, md, sec, person_name)
            contextualized = (
                f"{provenance_line}\n"
                f"{ctx}\n\n"
                f"[Section: {sec.breadcrumb}]\n\n"
                f"{sec.content}"
            )
            text_chunks.append(
                TextChunk(
                    text=contextualized,
                    index=sec.order,
                    metadata={
                        "breadcrumb": sec.breadcrumb,
                        "context": ctx,
                        "source": project.repo,
                        "project": label,
                        "kind": "github_readme",
                        "ingested_on": today,
                    },
                )
            )

        chunks = TextChunks(chunks=text_chunks)
        document_info = DocumentInfo(
            path=project.url,
            metadata={"source": project.repo, "kind": "github_readme"},
        )

        chunks = await text_chunk_embedder.run(text_chunks=chunks)

        graph = await extractor.run(
            chunks=chunks,
            document_info=document_info,
            schema=schema,
            lexical_graph_config=lexical_graph_config,
        )

        _normalize_node_names(graph)

        await writer.run(graph=graph, lexical_graph_config=lexical_graph_config)
        print(f"  Graph: {len(graph.nodes)} nodes, {len(graph.relationships)} rels")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_node_names(graph) -> None:
    for node in graph.nodes:
        name = node.properties.get("name")
        if isinstance(name, str):
            node.properties["name"] = name.strip().title()


def _clear_graph(driver: neo4j.Driver) -> None:
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Graph cleared.")


def _anchor_person(driver: neo4j.Driver, person_name: str) -> None:
    """Pre-create the canonical Person node with both :Person and :__Entity__
    labels so the post-ingestion resolver can unify it with extractor outputs."""
    with driver.session() as session:
        session.run(
            """
            MERGE (p:Person:__Entity__ {name: $name})
            SET p.is_canonical = true
            """,
            {"name": person_name.strip().title()},
        )


async def run_ingestion() -> None:
    driver = neo4j.GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )
    _clear_graph(driver)

    llm = AzureOpenAILLM(
        model_name="gpt-5.4",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-5.4",
        api_version="2024-12-01-preview",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        model_params={"temperature": 0.0},
    )
    embedder = AzureOpenAIEmbeddings(
        model="text-embedding-3-large",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="text-embedding-3-large",
        api_version="2024-12-01-preview",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        dimensions=3072,
    )

    # Strict schema: forbid any node/relationship/pattern outside the declared lists.
    schema_builder = SchemaBuilder()
    schema = await schema_builder.run(
        node_types=NODE_TYPES,
        relationship_types=RELATIONSHIP_TYPES,
        patterns=PATTERNS,
        additional_node_types=False,
        additional_relationship_types=False,
        additional_patterns=False,
    )

    text_chunk_embedder = TextChunkEmbedder(embedder=embedder)
    lexical_graph_config = LexicalGraphConfig()
    extractor = LLMEntityRelationExtractor(
        llm=llm,
        on_error=OnError.IGNORE,
        create_lexical_graph=True,
        use_structured_output=False,
    )
    writer = Neo4jWriter(driver=driver)

    today = date.today().strftime("%Y-%m-%d")

    # person_name is extracted from the first PDF. Initialise with an env-var
    # fallback so GitHub ingestion can proceed even if there are no PDFs yet.
    person_name: str = os.getenv("KG_PERSON_NAME", "Simone Bitti")
    _anchor_person(driver, person_name)

    for filename in os.listdir(FILES_FOLDER):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(FILES_FOLDER, filename)
        print(f"\n=== {filename} ===")

        md = pdf_to_markdown(filepath)
        print(f"  Markdown: {len(md)} chars")

        sections = split_markdown(md)
        print(f"  Sections: {len(sections)}")

        person_name = await extract_person_name(llm, md)
        print(f"  Person:   {person_name}")
        _anchor_person(driver, person_name)

        provenance_line = (
            f"This is a chunk extracted from the document \"{filename}\", "
            f"which refers to {person_name.lower()}. "
            f"IMPORTANT: the Person node for \"{person_name}\" already exists — "
            f"attach extracted relations to it, do NOT create a new Person node "
            f"for the CV owner."
        )

        # Build TextChunks: text = provenance + LLM context + breadcrumb + content
        text_chunks: list[TextChunk] = []
        for sec in sections:
            ctx = await contextualize(llm, md, sec)
            contextualized = (
                f"{provenance_line}\n"
                f"{ctx}\n\n"
                f"[Section: {sec.breadcrumb}]\n\n"
                f"{sec.content}"
            )
            text_chunks.append(
                TextChunk(
                    text=contextualized,
                    index=sec.order,
                    metadata={
                        "breadcrumb": sec.breadcrumb,
                        "context": ctx,
                        "source": filename,
                        "person": person_name,
                        "ingested_on": today,
                    },
                )
            )
        chunks = TextChunks(chunks=text_chunks)
        document_info = DocumentInfo(path=filepath, metadata={"source": filename})

        # 1) Embed contextualized chunk text (sets metadata['embedding'])
        chunks = await text_chunk_embedder.run(text_chunks=chunks)

        # 2) Strict-schema entity/relation extraction.
        #    With create_lexical_graph=True, the result already includes
        #    :Document and :Chunk nodes plus FROM_DOCUMENT / FROM_CHUNK edges.
        graph = await extractor.run(
            chunks=chunks,
            document_info=document_info,
            schema=schema,
            lexical_graph_config=lexical_graph_config,
        )

        # 3) Normalize names before writing to avoid case-variant duplicates
        _normalize_node_names(graph)

        # 4) Persist in one shot
        await writer.run(graph=graph, lexical_graph_config=lexical_graph_config)

        print(f"  Graph:    {len(graph.nodes)} nodes, {len(graph.relationships)} rels")

    # GitHub README ingestion — runs after PDFs so the Person node already exists
    # and person_name has been extracted from the CV.
    await run_github_ingestion(
        driver=driver,
        llm=llm,
        embedder=embedder,
        person_name=person_name,
        github_token=os.getenv("GITHUB_TOKEN"),
    )

    # Entity resolution: merge :__Entity__ nodes that share the same `name`
    # (per-label). This collapses duplicates produced across chunks where the
    # extractor emitted different random ids for the same real-world entity.
    resolver = SinglePropertyExactMatchResolver(driver=driver, resolve_property="name")
    stats = await resolver.run()
    print(f"\nEntity resolution: {stats}")

    resolver = AzureEmbeddingResolver(
        driver=driver,
        embedder=embedder,
        resolve_properties=["name"])
    resolver_stats = await resolver.run()
    print(f"\nEntity resolution: {resolver_stats}")

    with driver.session() as session:
        residual = session.run(
            """
            MATCH (n:__Entity__)
            WHERE n.name IS NOT NULL
            WITH labels(n) AS lbls, n.name AS name, count(*) AS c
            WHERE c > 1
            RETURN lbls, name, c ORDER BY c DESC LIMIT 10
            """
        ).data()
        if residual:
            print(f"WARNING: residual duplicates after resolver: {residual}")
        else:
            print("No residual duplicates.")

    driver.close()
    print("\nIngestion complete.")


if __name__ == "__main__":
    asyncio.run(run_ingestion())
