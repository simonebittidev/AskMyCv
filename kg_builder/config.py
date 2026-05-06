"""Centralised configuration: env-vars + YAML project list.

A single :class:`Settings` object is the only place where environment access
happens. Modules receive it via dependency injection — no scattered ``os.getenv``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv


# --- Constants ---------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF_FOLDER = REPO_ROOT / "files" / "Simone Bitti"
DEFAULT_PROJECTS_CONFIG = REPO_ROOT / "config" / "projects.yml"

# Owner of the KG (the person the CV is about).
DEFAULT_PERSON_NAME = "Simone Bitti"

# Azure OpenAI defaults — used only if the matching env-var is unset.
# Override at deploy time via:
#   AZURE_OPENAI_LLM_DEPLOYMENT
#   AZURE_OPENAI_EMBEDDING_DEPLOYMENT
#   AZURE_OPENAI_EMBEDDING_DIMENSIONS
#   AZURE_OPENAI_API_VERSION
_DEFAULT_LLM_DEPLOYMENT = "gpt-4.1-mini"
_DEFAULT_EMBEDDING_DEPLOYMENT = "text-embedding-3-large"
_DEFAULT_EMBEDDING_DIMENSIONS = 3072
_DEFAULT_OPENAI_API_VERSION = "2024-12-01-preview"


def _env(name: str, default: str) -> str:
    """Resolve an env-var with a fallback. Importing this at module scope means
    the values are read once at process start, which is what we want for Azure
    deployment names (they don't change while the service runs)."""
    # Load .env on first call so callers that import this module before
    # ``get_settings()`` still see the values.
    load_dotenv()
    return os.getenv(name, default)


# Module-level constants (read once at import time). Modules that need these
# values can ``from kg_builder.config import LLM_DEPLOYMENT`` as before; the
# value now reflects the env-var override.
LLM_DEPLOYMENT = _env("AZURE_OPENAI_LLM_DEPLOYMENT", _DEFAULT_LLM_DEPLOYMENT)
EMBEDDING_DEPLOYMENT = _env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", _DEFAULT_EMBEDDING_DEPLOYMENT)
EMBEDDING_DIMENSIONS = int(_env("AZURE_OPENAI_EMBEDDING_DIMENSIONS", str(_DEFAULT_EMBEDDING_DIMENSIONS)))
OPENAI_API_VERSION = _env("AZURE_OPENAI_API_VERSION", _DEFAULT_OPENAI_API_VERSION)


# --- Project list (YAML) -----------------------------------------------------


@dataclass(frozen=True)
class ProjectRef:
    """One curated GitHub project, as listed in ``config/projects.yml``."""

    repo: str  # "owner/repo"
    branch: Optional[str] = None
    display_name: Optional[str] = None

    @property
    def owner(self) -> str:
        return self.repo.split("/", 1)[0]

    @property
    def name(self) -> str:
        return self.repo.split("/", 1)[1]


def load_projects(path: Path = DEFAULT_PROJECTS_CONFIG) -> List[ProjectRef]:
    """Load the curated project list. Returns an empty list if the file is missing."""
    import logging
    log = logging.getLogger(__name__)

    if not path.exists():
        log.warning("[config] projects file NOT found at %s", path.resolve())
        return []
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    projects = raw.get("projects", []) or []
    refs = [
        ProjectRef(
            repo=p["repo"],
            branch=p.get("branch"),
            display_name=p.get("display_name"),
        )
        for p in projects
        if "repo" in p
    ]
    log.info("[config] loaded %d project(s) from %s", len(refs), path)
    return refs


# --- Settings ----------------------------------------------------------------


@dataclass
class Settings:
    """Runtime configuration assembled from env-vars and YAML."""

    # Neo4j
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str

    # Azure OpenAI (optional: SDK reads them automatically from env)
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_api_version: str = OPENAI_API_VERSION
    azure_llm_deployment: str = LLM_DEPLOYMENT
    azure_embedding_deployment: str = EMBEDDING_DEPLOYMENT
    azure_embedding_dimensions: int = EMBEDDING_DIMENSIONS

    # GitHub
    github_token: Optional[str] = None
    github_username: Optional[str] = None

    # Filesystem inputs
    pdf_folder: Path = DEFAULT_PDF_FOLDER
    projects_config: Path = DEFAULT_PROJECTS_CONFIG

    # Behaviour
    person_name: str = DEFAULT_PERSON_NAME
    enable_communities: bool = True

    # Resolved at load time
    projects: List[ProjectRef] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        missing = [k for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD") if not os.getenv(k)]
        if missing:
            raise RuntimeError(f"Missing required env-vars: {', '.join(missing)}")

        return cls(
            neo4j_uri=os.environ["NEO4J_URI"],
            neo4j_username=os.environ["NEO4J_USERNAME"],
            neo4j_password=os.environ["NEO4J_PASSWORD"],
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_openai_api_version=_env("AZURE_OPENAI_API_VERSION", _DEFAULT_OPENAI_API_VERSION),
            azure_llm_deployment=_env("AZURE_OPENAI_LLM_DEPLOYMENT", _DEFAULT_LLM_DEPLOYMENT),
            azure_embedding_deployment=_env(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", _DEFAULT_EMBEDDING_DEPLOYMENT
            ),
            azure_embedding_dimensions=int(
                _env("AZURE_OPENAI_EMBEDDING_DIMENSIONS", str(_DEFAULT_EMBEDDING_DIMENSIONS))
            ),
            github_token=os.getenv("GITHUB_TOKEN"),
            github_username=os.getenv("GITHUB_USERNAME"),
            pdf_folder=Path(os.getenv("KG_PDF_FOLDER", str(DEFAULT_PDF_FOLDER))),
            projects_config=Path(os.getenv("KG_PROJECTS_CONFIG", str(DEFAULT_PROJECTS_CONFIG))),
            person_name=os.getenv("KG_PERSON_NAME", DEFAULT_PERSON_NAME),
            enable_communities=os.getenv("KG_ENABLE_COMMUNITIES", "true").lower() == "true",
            projects=load_projects(Path(os.getenv("KG_PROJECTS_CONFIG", str(DEFAULT_PROJECTS_CONFIG)))),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor — call this from the pipeline / app.py."""
    return Settings.from_env()
