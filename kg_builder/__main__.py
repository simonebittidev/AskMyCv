"""Entry point: ``python -m kg_builder`` rebuilds the knowledge graph end-to-end."""

import asyncio

from kg_builder.pipeline import run_pipeline


if __name__ == "__main__":
    asyncio.run(run_pipeline())
