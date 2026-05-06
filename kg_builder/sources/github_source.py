"""GitHub source: fetches README + metadata for the curated project list."""

from __future__ import annotations

import base64
import logging
import re
from typing import List, Optional

import httpx

from kg_builder.config import ProjectRef
from kg_builder.models import ProjectDoc, SourceDocument
from kg_builder.sources.base import DocumentSource

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"
_DEFAULT_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "ChatMyCv-kg-builder",
}

# Regex to strip noisy README artefacts before passing to the LLM:
# - Markdown images: ![alt](url)
# - HTML <img> tags
# - Shield/badge URLs (img.shields.io, badgen.net, ...)
_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_IMAGE_HTML_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_BADGE_LINK_RE = re.compile(
    r"\[!\[[^\]]*\]\((?:https?://(?:img\.shields\.io|badgen\.net|badge\.fury\.io)[^)]+)\)\]\([^)]+\)"
)


def _clean_readme(md: str) -> str:
    """Remove badges and inline images that add noise without semantic value."""
    md = _BADGE_LINK_RE.sub("", md)
    md = _IMAGE_MD_RE.sub("", md)
    md = _IMAGE_HTML_RE.sub("", md)
    # Collapse 3+ blank lines created by the substitutions.
    return re.sub(r"\n{3,}", "\n\n", md).strip()


class GitHubSource(DocumentSource):
    """Loads README + repo metadata for every project in the curated list."""

    def __init__(
        self,
        projects: List[ProjectRef],
        token: Optional[str] = None,
        timeout: float = 15.0,
    ) -> None:
        self.projects = projects
        self.token = token
        self.timeout = timeout

    async def load(self) -> List[SourceDocument]:
        if not self.projects:
            logger.warning(
                "[github] no projects configured — check that config/projects.yml exists "
                "and that Settings.projects is populated"
            )
            return []

        logger.info(
            "[github] %d project(s) to fetch: %s",
            len(self.projects),
            [p.repo for p in self.projects],
        )

        headers = dict(_DEFAULT_HEADERS)
        if self.token:
            # Last-4 char fingerprint to confirm a token is loaded without leaking it.
            logger.info("[github] using token ending in '...%s'", self.token[-4:])
            headers["Authorization"] = f"Bearer {self.token}"
        else:
            logger.warning(
                "[github] no GITHUB_TOKEN provided — rate limit is 60 req/h and "
                "private repos will be invisible"
            )

        async with httpx.AsyncClient(headers=headers, timeout=self.timeout) as client:
            await self._log_rate_limit(client)
            project_docs = []
            for ref in self.projects:
                try:
                    doc = await self._fetch_one(client, ref)
                    if doc is not None:
                        project_docs.append(doc)
                except Exception as exc:
                    logger.exception("[github] failed to fetch %s: %s", ref.repo, exc)

        logger.info(
            "[github] fetched %d/%d projects successfully",
            len(project_docs),
            len(self.projects),
        )
        return [self._to_source_document(p) for p in project_docs if p]

    async def _log_rate_limit(self, client: httpx.AsyncClient) -> None:
        try:
            resp = await client.get(f"{_GITHUB_API}/rate_limit")
            if resp.status_code == 200:
                core = resp.json().get("resources", {}).get("core", {})
                logger.info(
                    "[github] rate-limit: %s/%s remaining (resets in %ss)",
                    core.get("remaining"),
                    core.get("limit"),
                    max(0, (core.get("reset", 0) or 0) - 0),
                )
            else:
                logger.warning("[github] rate-limit probe HTTP %s: %s",
                               resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.warning("[github] rate-limit probe failed: %s", exc)

    # ----- HTTP --------------------------------------------------------------

    async def _fetch_one(self, client: httpx.AsyncClient, ref: ProjectRef) -> Optional[ProjectDoc]:
        meta_url = f"{_GITHUB_API}/repos/{ref.repo}"
        readme_url = f"{_GITHUB_API}/repos/{ref.repo}/readme"
        if ref.branch:
            readme_url += f"?ref={ref.branch}"

        logger.debug("[github] GET %s", meta_url)
        meta_resp = await client.get(meta_url)
        if meta_resp.status_code != 200:
            logger.warning(
                "[github] %s metadata HTTP %s — body: %s",
                ref.repo,
                meta_resp.status_code,
                meta_resp.text[:300],
            )
            if meta_resp.status_code == 404:
                logger.warning(
                    "[github] 404 usually means: wrong owner/name, private repo without "
                    "token scope, or token lacks 'repo' / 'public_repo' permission"
                )
            return None
        meta = meta_resp.json()

        logger.debug("[github] GET %s", readme_url)
        readme_resp = await client.get(readme_url)
        if readme_resp.status_code != 200:
            logger.warning(
                "[github] %s README HTTP %s — body: %s",
                ref.repo,
                readme_resp.status_code,
                readme_resp.text[:300],
            )
            readme_md = ""
        else:
            payload = readme_resp.json()
            readme_md = base64.b64decode(payload.get("content", "")).decode("utf-8", errors="replace")
        readme_md = _clean_readme(readme_md)

        logger.info(
            "[github] fetched %s (%d README chars, language=%s, stars=%s)",
            ref.repo,
            len(readme_md),
            meta.get("language"),
            meta.get("stargazers_count"),
        )

        return ProjectDoc(
            repo=meta.get("full_name", ref.repo),
            name=meta.get("name", ref.name),
            display_name=ref.display_name,
            url=meta.get("html_url", f"https://github.com/{ref.repo}"),
            description=meta.get("description"),
            homepage=meta.get("homepage") or None,
            primary_language=meta.get("language"),
            topics=meta.get("topics") or [],
            stars=meta.get("stargazers_count", 0),
            pushed_at=meta.get("pushed_at"),
            readme_md=readme_md,
        )

    # ----- conversion --------------------------------------------------------

    @staticmethod
    def _to_source_document(p: ProjectDoc) -> SourceDocument:
        # Build a Markdown header so the LLM gets the metadata in-context.
        header_lines = [
            f"# {p.display_name or p.name}",
            "",
            f"- **Repository:** {p.url}",
        ]
        if p.description:
            header_lines.append(f"- **Description:** {p.description}")
        if p.primary_language:
            header_lines.append(f"- **Primary language:** {p.primary_language}")
        if p.topics:
            header_lines.append(f"- **Topics:** {', '.join(p.topics)}")
        if p.homepage:
            header_lines.append(f"- **Homepage:** {p.homepage}")
        header_lines.append("")
        body = "\n".join(header_lines) + "\n" + (p.readme_md or "")

        return SourceDocument(
            content=body,
            source_id=p.repo,
            kind="github_readme",
            metadata=p.model_dump(mode="json"),
        )
