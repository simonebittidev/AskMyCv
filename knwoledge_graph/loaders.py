import os
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional
import neo4j
from helpers import _anchor_person, contextualize, contextualize_github, extract_person_name, pdf_to_markdown, split_markdown
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.experimental.components.schema import (
    NodeType as SchemaNodeType,
    RelationshipType as SchemaRelationshipType,
)
from neo4j_graphrag.experimental.components.types import (
    DocumentInfo,
    TextChunk,
    TextChunks,
)
from schema import NODE_TYPES, PATTERNS, RELATIONSHIP_TYPES, GITHUB_NODE_TYPES, GITHUB_PATTERNS, GITHUB_RELATIONSHIP_TYPES
from github_source import fetch_github_projects, load_project_refs

@dataclass
class SourceBatch:
    chunks: TextChunks
    document_info: DocumentInfo


class SourceLoader(ABC):
    @property
    @abstractmethod
    def node_types(self) -> list[SchemaNodeType]: ...

    @property
    @abstractmethod
    def relationship_types(self) -> list[SchemaRelationshipType]: ...

    @property
    @abstractmethod
    def patterns(self) -> list[tuple[str, str, str]]: ...

    @abstractmethod
    def iter_batches(self) -> AsyncGenerator[SourceBatch, None]: ...


class PDFLoader(SourceLoader):
    def __init__(self, folder: str, llm: AzureOpenAILLM, driver: neo4j.Driver) -> None:
        self.folder = folder
        self.llm = llm
        self.driver = driver
        self.person_name: str = os.getenv("KG_PERSON_NAME", "Simone Bitti")

    @property
    def node_types(self) -> list[SchemaNodeType]:
        return NODE_TYPES

    @property
    def relationship_types(self) -> list[SchemaRelationshipType]:
        return RELATIONSHIP_TYPES

    @property
    def patterns(self) -> list[tuple[str, str, str]]:
        return PATTERNS

    async def iter_batches(self) -> AsyncGenerator[SourceBatch, None]:
        today = date.today().strftime("%Y-%m-%d")
        _anchor_person(self.driver, self.person_name)

        for filename in sorted(os.listdir(self.folder)):
            if not filename.lower().endswith(".pdf"):
                continue

            filepath = os.path.join(self.folder, filename)
            print(f"\n=== {filename} ===")

            md = pdf_to_markdown(filepath)
            print(f"  Markdown: {len(md)} chars")

            sections = split_markdown(md)
            print(f"  Sections: {len(sections)}")

            self.person_name = await extract_person_name(self.llm, md)
            print(f"  Person:   {self.person_name}")
            _anchor_person(self.driver, self.person_name)

            provenance_line = (
                f"This is a chunk extracted from the document \"{filename}\", "
                f"which refers to {self.person_name.lower()}. "
            )

            text_chunks: list[TextChunk] = []
            for sec in sections:
                ctx = await contextualize(self.llm, md, sec)
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
                            "person": self.person_name,
                            "ingested_on": today,
                        },
                    )
                )

            yield SourceBatch(
                chunks=TextChunks(chunks=text_chunks),
                document_info=DocumentInfo(path=filepath, metadata={"source": filename}),
            )


class GitHubLoader(SourceLoader):
    def __init__(
        self,
        llm: AzureOpenAILLM,
        person_name: str,
        github_token: Optional[str] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        self.person_name = person_name
        self.github_token = github_token
        self.config_path = config_path or Path(__file__).parent.parent / "config" / "projects.yml"

    @property
    def node_types(self) -> list[SchemaNodeType]:
        return GITHUB_NODE_TYPES

    @property
    def relationship_types(self) -> list[SchemaRelationshipType]:
        return GITHUB_RELATIONSHIP_TYPES

    @property
    def patterns(self) -> list[tuple[str, str, str]]:
        return GITHUB_PATTERNS

    async def iter_batches(self) -> AsyncGenerator[SourceBatch, None]:

        refs = load_project_refs(self.config_path)
        if not refs:
            print("[github] no projects configured — check config/projects.yml")
            return

        projects = await fetch_github_projects(refs, token=self.github_token)
        if not projects:
            print("[github] no projects fetched")
            return

        today = date.today().strftime("%Y-%m-%d")

        for project in projects:
            label = project.display_name or project.name
            print(f"\n=== GitHub: {label} ({project.repo}) ===")

            md = project.to_markdown()
            sections = split_markdown(md)
            print(f"  README sections: {len(sections)}")

            provenance_line = (
                f"This chunk comes from the GitHub README of \"{label}\" "
                f"(repository: {project.repo}), a personal project created by {self.person_name}. "
            )

            text_chunks: list[TextChunk] = []
            for sec in sections:
                ctx = await contextualize_github(self.llm, md, sec, self.person_name)
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

            yield SourceBatch(
                chunks=TextChunks(chunks=text_chunks),
                document_info=DocumentInfo(
                    path=project.url,
                    metadata={"source": project.repo, "kind": "github_readme"},
                ),
            )
