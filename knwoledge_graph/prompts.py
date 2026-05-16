# ─────────────────────────────────────────────────────────────────────────────
# Helpers — Contextual chunking
# ─────────────────────────────────────────────────────────────────────────────

CONTEXT_PROMPT = """You are an expert at situating excerpts within their source document.

You will be given the full document and one of its chunks. Write 2-3 short sentences that situate the chunk inside the document, so a reader (or a retrieval system) can understand what this chunk is about and how it fits in the bigger picture.

Rules:
- Output ONLY the contextualization sentences. No preamble, no quotes, no bullets.
- Do not summarize the whole document — only what's needed to place this chunk.
- Use the same language as the document.
- Keep it under 60 words."""

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
