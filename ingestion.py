"""
Ingestion pipeline:
  PDF -> Markdown (pymupdf4llm)
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
from typing import List

import neo4j
import pymupdf4llm
from dotenv import load_dotenv
from pydantic import BaseModel

from neo4j_graphrag.embeddings.openai import AzureOpenAIEmbeddings as _AzureOpenAIEmbeddings


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
    SchemaNodeType(label="Person",              properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="Role",                properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="Organization",        properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="Location",            properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="country", type="STRING")]),
    # — IT-specific technical nodes
    SchemaNodeType(label="ProgrammingLanguage", properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="Framework",           properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="Database",            properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="CloudPlatform",       properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="DevOpsTool",          properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="Technology",          properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="Domain",              properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="Methodology",         properties=[PropertyType(name="name",    type="STRING")]),
    # — Skills & languages
    SchemaNodeType(label="Skill",               properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="Language",            properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="Proficiency",         properties=[PropertyType(name="name",    type="STRING")]),
    # — Education & credentials
    SchemaNodeType(label="Degree",              properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="Certification",       properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="Award",               properties=[PropertyType(name="name",    type="STRING")]),
    # — Projects & activities
    SchemaNodeType(label="Project",             properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="PersonalProject",     properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="ProjectUrl",          properties=[PropertyType(name="name",    type="STRING")]),
    SchemaNodeType(label="Activity",            properties=[PropertyType(name="name",    type="STRING")]),
    # — Contact & time
    SchemaNodeType(label="Contact",             properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="type", type="STRING")]),
    SchemaNodeType(label="DateRange",           properties=[PropertyType(name="name",    type="STRING")]),
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
    ),
    SchemaRelationshipType(
        label="WORKED_AT",
        description=(
            f"PAST employment (Person→Organization). Use when the end date is explicitly BEFORE today ({_TODAY}). "
            "Default choice when there is any doubt about whether the role is still active."
        ),
    ),
    SchemaRelationshipType(label="HAS_ROLE",        description="Links a Person to a Role they hold or held (e.g. 'Software Engineer', 'Data Scientist')."),
    SchemaRelationshipType(label="ROLE_AT",         description="Links a Role to the Organization where it was or is held."),
    SchemaRelationshipType(label="HAS_SKILL",       description="Links a Person to a Skill (technical or soft skill)."),
    SchemaRelationshipType(label="HAS_PROFICIENCY", description="Links a Skill or Language node to a Proficiency level (e.g. Beginner, Intermediate, Advanced, Fluent, Native)."),
    SchemaRelationshipType(label="KNOWS_LANGUAGE",  description="A Person knows a natural Language (Italian, English, French, ...)."),
    SchemaRelationshipType(label="HAS_CERTIFICATION", description="A Person holds a Certification."),
    SchemaRelationshipType(label="HAS_AWARD",       description="A Person received an Award or recognition."),
    SchemaRelationshipType(label="STUDIED_AT",      description="A Person studied at an Organization (university, school, bootcamp)."),
    SchemaRelationshipType(label="OBTAINED_DEGREE", description="A Person obtained a Degree (e.g. 'Bachelor of Science in Computer Science')."),
    SchemaRelationshipType(label="GRANTED_BY",      description="A Certification, Award, or Degree was issued/granted by an Organization."),
    SchemaRelationshipType(label="HAS_CONTACT",     description="A Person has a Contact (email, phone, LinkedIn, GitHub, website). Set the `type` property on the Contact node."),
    SchemaRelationshipType(label="INVOLVED_IN",     description="A Person was involved in a Project or PersonalProject."),
    SchemaRelationshipType(label="PARTICIPATED_IN", description="A Person participated in an Activity (volunteering, sport, association, event)."),
    SchemaRelationshipType(label="LOCATED_IN",      description="A Person or Organization is located in a Location (city, country, region)."),
    SchemaRelationshipType(label="USES",            description="A Role, Project, or PersonalProject uses a Technology, Framework, Database, or DevOpsTool. Use more specific relations (DEPLOYED_ON, APPLIES) when applicable."),
    SchemaRelationshipType(label="DEPLOYED_ON",     description="A Role, Project, or PersonalProject is deployed on or hosted by a CloudPlatform (e.g. AWS, Azure, GCP)."),
    SchemaRelationshipType(label="SPECIALIZES_IN",  description="A Person or Role specializes in a Domain (e.g. Backend, Machine Learning, Data Engineering, NLP, DevOps, Frontend)."),
    SchemaRelationshipType(label="APPLIES",         description="A Role or Project applies a Methodology or practice (e.g. Agile, Scrum, TDD, REST, Microservices, CI/CD)."),
    SchemaRelationshipType(label="DURING",          description="Links a Role, Project, PersonalProject, Certification, Degree, Award, or Activity to its DateRange (e.g. '2020–2023', 'Jan 2021 – Present')."),
    SchemaRelationshipType(label="HAS_URL",         description="Links a Project or PersonalProject to a ProjectUrl."),
    SchemaRelationshipType(label="PART_OF",         description="Links a Project to the Organization it was developed for or within."),
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
# Step 1 — PDF to Markdown
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_markdown(pdf_path: str) -> str:
    return pymupdf4llm.to_markdown(pdf_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Markdown chunking (header-aware, paragraph fallback)
# ─────────────────────────────────────────────────────────────────────────────

class MarkdownSection(BaseModel):
    breadcrumb: str
    content: str
    order: int


HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def split_markdown(md: str, max_chars: int = 1800) -> List[MarkdownSection]:
    lines = md.splitlines()
    raw_sections: list[tuple[list[str], list[str]]] = []
    header_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []

    def flush():
        if current_lines and any(l.strip() for l in current_lines):
            raw_sections.append(([t for _, t in header_stack], current_lines.copy()))

    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            flush()
            current_lines = []
            level = len(m.group(1))
            title = m.group(2).strip()
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, title))
        else:
            current_lines.append(line)
    flush()

    if not raw_sections:
        raw_sections = [([], lines)]

    out: list[MarkdownSection] = []
    order = 0
    for path, content_lines in raw_sections:
        breadcrumb = " > ".join(path) if path else "(root)"
        body = "\n".join(content_lines).strip()
        if not body:
            continue

        if len(body) <= max_chars:
            out.append(MarkdownSection(breadcrumb=breadcrumb, content=body, order=order))
            order += 1
            continue

        paragraphs = re.split(r"\n\s*\n", body)
        buf, buf_len = [], 0
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if buf and buf_len + len(p) + 2 > max_chars:
                out.append(MarkdownSection(breadcrumb=breadcrumb, content="\n\n".join(buf), order=order))
                order += 1
                buf, buf_len = [], 0
            buf.append(p)
            buf_len += len(p) + 2
        if buf:
            out.append(MarkdownSection(breadcrumb=breadcrumb, content="\n\n".join(buf), order=order))
            order += 1

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


async def run_ingestion() -> None:
    driver = neo4j.GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )
    _clear_graph(driver)

    llm = AzureOpenAILLM(
        model_name="gpt-5.4-mini",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-5.4-mini",
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

        provenance_line = (
            f"This is a chunk extracted from the document \"{filename}\", "
            f"which refers to {person_name.lower()}."
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

    # Entity resolution: merge :__Entity__ nodes that share the same `name`
    # (per-label). This collapses duplicates produced across chunks where the
    # extractor emitted different random ids for the same real-world entity.
    resolver = SinglePropertyExactMatchResolver(driver=driver, resolve_property="name")
    stats = await resolver.run()
    print(f"\nEntity resolution: {stats}")

    driver.close()
    print("\nIngestion complete.")


if __name__ == "__main__":
    asyncio.run(run_ingestion())
