"""PDF source: parses CV / Cover Letter PDFs straight to Markdown via Docling.

Docling (IBM, 2024-2025) preserves layout, tables, lists and headings without
relying on a vision LLM, so parsing is deterministic, fast, and free.

If Docling is unavailable or fails on a file, we fall back to a vision-LLM
based parser that converts pages to Markdown (not HTML). This keeps the
pipeline robust on edge-case PDFs.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import List, Optional

from kg_builder.models import SourceDocument
from kg_builder.sources.base import DocumentSource

logger = logging.getLogger(__name__)


# Map filename hints to a kind the rest of the pipeline understands.
_KIND_HINTS = (
    ("cover", "cover_letter"),
    ("motivation", "cover_letter"),
    ("cv", "cv"),
    ("resume", "cv"),
)


def _infer_kind(filename: str) -> str:
    name = filename.lower()
    for hint, kind in _KIND_HINTS:
        if hint in name:
            return kind
    # Unknown PDFs default to "cv" — they still go through the no-projects schema.
    return "cv"


class PdfSource(DocumentSource):
    """Convert every ``*.pdf`` in a folder to a :class:`SourceDocument`."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)

    async def load(self) -> List[SourceDocument]:
        if not self.folder.exists():
            logger.warning("[pdf] folder %s does not exist", self.folder)
            return []

        pdfs = sorted(p for p in self.folder.iterdir() if p.suffix.lower() == ".pdf")
        if not pdfs:
            logger.info("[pdf] no PDFs found in %s", self.folder)
            return []

        # Run conversions concurrently — Docling parsing is CPU-bound, but PDFs
        # are typically few (CV + cover letter), so a thread executor is fine.
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, self._convert_one, pdf) for pdf in pdfs]
        results = await asyncio.gather(*tasks)
        return [doc for doc in results if doc is not None]

    # ----- conversion helpers ------------------------------------------------

    def _convert_one(self, path: Path) -> Optional[SourceDocument]:
        markdown = _convert_with_docling(path)
        if markdown is None:
            logger.warning("[pdf] Docling failed for %s, falling back to vision LLM", path.name)
            markdown = _convert_with_vision_llm(path)
        if not markdown:
            logger.error("[pdf] could not parse %s", path.name)
            return None

        kind = _infer_kind(path.name)
        logger.info("[pdf] parsed %s as %s (%d chars)", path.name, kind, len(markdown))
        return SourceDocument(
            content=markdown,
            source_id=path.name,
            kind=kind,
            metadata={"path": str(path)},
        )


def _convert_with_docling(path: Path) -> Optional[str]:
    """Primary parser. Returns ``None`` on any failure so the caller can fall back."""
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
    except ImportError:
        logger.warning("[pdf] docling not installed; install with `pip install docling`")
        return None

    try:
        converter = DocumentConverter()
        result = converter.convert(str(path))
        return result.document.export_to_markdown()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("[pdf] Docling error on %s: %s", path.name, exc)
        return None


def _convert_with_vision_llm(path: Path) -> Optional[str]:
    """Fallback: render pages to PNG and ask the vision LLM for Markdown."""
    try:
        import fitz  # PyMuPDF
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import AzureChatOpenAI
    except ImportError as exc:
        logger.error("[pdf] vision-LLM fallback dependencies missing: %s", exc)
        return None

    from kg_builder.config import LLM_DEPLOYMENT, OPENAI_API_VERSION

    images_b64: List[str] = []
    try:
        doc = fitz.open(str(path))
        for page in doc:
            pix = page.get_pixmap()
            images_b64.append(base64.b64encode(pix.tobytes(output="png")).decode("ascii"))
    except Exception as exc:
        logger.exception("[pdf] PyMuPDF failed for %s: %s", path.name, exc)
        return None

    if not images_b64:
        return None

    system_prompt = (
        "You are an expert at reading scanned PDF pages of CVs and cover letters. "
        "Convert every page into a single, well-structured **Markdown** document. "
        "Preserve hierarchy: use ATX headings (# / ## / ###) for sections, bullet "
        "lists for skills/items, and proper Markdown tables when applicable. "
        "Do not add commentary — output only Markdown."
    )

    messages = [SystemMessage(content=system_prompt)]
    for image in images_b64:
        messages.append(
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    }
                ]
            )
        )

    llm = AzureChatOpenAI(
        azure_deployment=LLM_DEPLOYMENT,
        openai_api_version=OPENAI_API_VERSION,
        temperature=0.0,
    )
    try:
        chain = ChatPromptTemplate.from_messages(messages) | llm
        response = chain.invoke({})
        text = response.content if hasattr(response, "content") else str(response)
        return text.replace("```markdown", "").replace("```", "").strip()
    except Exception as exc:
        logger.exception("[pdf] vision LLM failed for %s: %s", path.name, exc)
        return None
