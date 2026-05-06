"""Standalone diagnostic for the GitHub source.

Run with::

    python -m kg_builder.sources._debug_github

It loads ``config/projects.yml``, prints the resolved project list, hits the
GitHub API for each repo, and reports exactly what fails — without touching
Neo4j or the rest of the pipeline. Use this when ``run_pipeline`` reports
"no GitHub READMEs found".
"""

from __future__ import annotations

import asyncio
import logging
import sys

from kg_builder.config import get_settings
from kg_builder.sources.github_source import GitHubSource


async def _main() -> int:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = get_settings()

    print("=" * 70)
    print(f"projects_config = {settings.projects_config}")
    print(f"projects loaded = {len(settings.projects)}")
    for p in settings.projects:
        print(f"  - {p.repo}  (branch={p.branch}, display={p.display_name})")
    print(f"GITHUB_TOKEN    = {'set (...' + settings.github_token[-4:] + ')' if settings.github_token else 'MISSING'}")
    print("=" * 70)

    if not settings.projects:
        print(
            "\nNo projects loaded. Check that config/projects.yml exists and that "
            "it has a 'projects:' top-level key with at least one entry like:\n"
            "\n  projects:\n    - repo: simonebitti/ChatMyCv\n"
        )
        return 1

    source = GitHubSource(projects=settings.projects, token=settings.github_token)
    docs = await source.load()
    print("\nResults:")
    for d in docs:
        print(f"  - {d.source_id}: {len(d.content)} chars (kind={d.kind})")
    if not docs:
        print(
            "\nFetched 0 documents. Most common causes:\n"
            "  * 404 -> repo name typo (case-sensitive after the slash)\n"
            "  * 401/403 -> bad/missing GITHUB_TOKEN, or token lacks public_repo scope\n"
            "  * Network/proxy issue\n"
            "Re-read the [github] log lines above for the exact HTTP status."
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
