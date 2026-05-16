"""GitHub source: fetches README + metadata for the curated project list.

Reads ``config/projects.yml``, calls the GitHub REST API for each listed repo,
and returns a list of :class:`GitHubProject` objects ready for ingestion.

Badge/image noise is stripped from READMEs before they are passed to the LLM
to avoid wasting tokens on non-semantic content.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import httpx
import yaml

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"
_DEFAULT_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "ChatMyCv-kg-builder",
}

# Strip badges and inline images — they add no semantic value for the LLM.
_BADGE_LINK_RE = re.compile(
    r"\[!\[[^\]]*\]\((?:https?://(?:img\.shields\.io|badgen\.net|badge\.fury\.io)[^)]+)\)\]\([^)]+\)"
)
_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_IMAGE_HTML_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)


def _clean_readme(md: str) -> str:
    """Remove badges and inline images that add noise without semantic value."""
    md = _BADGE_LINK_RE.sub("", md)
    md = _IMAGE_MD_RE.sub("", md)
    md = _IMAGE_HTML_RE.sub("", md)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


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


@dataclass
class GitHubProject:
    """Normalised metadata + cleaned README for a single GitHub repository."""

    repo: str
    name: str
    display_name: Optional[str]
    url: str
    description: Optional[str]
    homepage: Optional[str]
    primary_language: Optional[str]
    topics: List[str]
    stars: int
    readme_md: str

    def to_markdown(self) -> str:
        """Return a Markdown document that combines a metadata header with the README body.

        The header gives the LLM structured context (repo URL, language, topics)
        before it reads the raw README, which improves extraction quality.
        """
        lines = [
            f"# {self.display_name or self.name}",
            "",
            f"- **Repository:** {self.url}",
        ]
        if self.description:
            lines.append(f"- **Description:** {self.description}")
        if self.primary_language:
            lines.append(f"- **Primary language:** {self.primary_language}")
        if self.topics:
            lines.append(f"- **Topics:** {', '.join(self.topics)}")
        if self.homepage:
            lines.append(f"- **Homepage:** {self.homepage}")
        lines.append("")
        return "\n".join(lines) + "\n" + (self.readme_md or "")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = Path(__file__).parent / "config" / "projects.yml"


def load_project_refs(config_path: Path = _DEFAULT_CONFIG) -> List[ProjectRef]:
    """Load the curated project list from YAML. Returns [] if the file is missing."""
    if not config_path.exists():
        logger.warning("[github] projects config not found at %s", config_path.resolve())
        return []
    with config_path.open("r", encoding="utf-8") as fh:
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
    logger.info("[github] loaded %d project(s) from %s", len(refs), config_path)
    return refs


# ---------------------------------------------------------------------------
# GitHub API fetcher
# ---------------------------------------------------------------------------


async def fetch_github_projects(
    projects: List[ProjectRef],
    token: Optional[str] = None,
    timeout: float = 15.0,
) -> List[GitHubProject]:
    """Fetch repo metadata and README for every :class:`ProjectRef` in *projects*.

    Args:
        projects: List of repos to fetch (from :func:`load_project_refs`).
        token: Optional GitHub personal access token. Without a token the API
               allows only 60 unauthenticated requests per hour and private
               repos are invisible.
        timeout: Per-request timeout in seconds.

    Returns:
        Successfully fetched projects. Repos that fail (404, network error, …)
        are logged and silently skipped so the rest of the pipeline continues.
    """
    if not projects:
        logger.warning("[github] no projects configured — check config/projects.yml")
        return []

    headers = dict(_DEFAULT_HEADERS)
    if token:
        logger.info("[github] using token ending in '...%s'", token[-4:])
        headers["Authorization"] = f"Bearer {token}"
    else:
        logger.warning(
            "[github] no GITHUB_TOKEN — rate limit is 60 req/h, private repos invisible"
        )

    results: List[GitHubProject] = []
    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        await _log_rate_limit(client)
        for ref in projects:
            try:
                project = await _fetch_one(client, ref)
                if project is not None:
                    results.append(project)
            except Exception as exc:
                logger.exception("[github] failed to fetch %s: %s", ref.repo, exc)

    logger.info(
        "[github] fetched %d/%d projects successfully",
        len(results),
        len(projects),
    )
    return results


async def _log_rate_limit(client: httpx.AsyncClient) -> None:
    try:
        resp = await client.get(f"{_GITHUB_API}/rate_limit")
        if resp.status_code == 200:
            core = resp.json().get("resources", {}).get("core", {})
            logger.info(
                "[github] rate-limit: %s/%s remaining",
                core.get("remaining"),
                core.get("limit"),
            )
    except Exception as exc:
        logger.warning("[github] rate-limit probe failed: %s", exc)


async def _fetch_one(
    client: httpx.AsyncClient, ref: ProjectRef
) -> Optional[GitHubProject]:
    meta_url = f"{_GITHUB_API}/repos/{ref.repo}"
    readme_url = f"{_GITHUB_API}/repos/{ref.repo}/readme"
    if ref.branch:
        readme_url += f"?ref={ref.branch}"

    meta_resp = await client.get(meta_url)
    if meta_resp.status_code != 200:
        logger.warning(
            "[github] %s metadata HTTP %s — %s",
            ref.repo,
            meta_resp.status_code,
            meta_resp.text[:200],
        )
        if meta_resp.status_code == 404:
            logger.warning(
                "[github] 404 = wrong owner/name, private repo without token, "
                "or token missing 'repo'/'public_repo' scope"
            )
        return None
    meta = meta_resp.json()

    readme_resp = await client.get(readme_url)
    if readme_resp.status_code == 200:
        payload = readme_resp.json()
        readme_md = base64.b64decode(payload.get("content", "")).decode(
            "utf-8", errors="replace"
        )
    else:
        logger.warning(
            "[github] %s README HTTP %s", ref.repo, readme_resp.status_code
        )
        readme_md = ""
    readme_md = _clean_readme(readme_md)

    logger.info(
        "[github] fetched %s (%d README chars, lang=%s, stars=%s)",
        ref.repo,
        len(readme_md),
        meta.get("language"),
        meta.get("stargazers_count"),
    )

    return GitHubProject(
        repo=meta.get("full_name", ref.repo),
        name=meta.get("name", ref.name),
        display_name=ref.display_name,
        url=meta.get("html_url", f"https://github.com/{ref.repo}"),
        description=meta.get("description"),
        homepage=meta.get("homepage") or None,
        primary_language=meta.get("language"),
        topics=meta.get("topics") or [],
        stars=meta.get("stargazers_count", 0),
        readme_md=readme_md,
    )
