"""Summarise a Markdown document into brief + detailed summaries."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from kg_builder.extraction._llm import get_chat_llm
from kg_builder.models import DocumentSummary

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are an expert at analysing Markdown documents extracted from CVs, cover \
letters, or technical READMEs.

## Goal
Read the full Markdown document and produce:
- A brief summary (2-3 sentences) describing the overall content and purpose.
- An exhaustive detailed summary covering every relevant fact about the \
  person or project: background, experience, skills, technologies, and any \
  other information present in the source.

## Instructions
- Analyse all sections, headings, bullet lists, tables and paragraphs.
- Use neutral, precise language. Do not rewrite, invent, translate or \
  reinterpret. Just summarise faithfully.
- Do not omit any significant information.
- If something is ambiguous, include it as best you can without fabricating.

## Output Format
Return only valid JSON matching the schema, no extra commentary."""


def summarize_document(markdown: str) -> DocumentSummary:
    """Summarise a Markdown document into a :class:`DocumentSummary`."""
    llm = get_chat_llm(temperature=0.0)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Here is the Markdown document to summarise:\n\n{markdown}"),
    ]
    return llm.with_structured_output(DocumentSummary).invoke(messages)
