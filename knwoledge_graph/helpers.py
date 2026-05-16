import asyncio
import base64
import os
from typing import List
import fitz  # pymupdf
import neo4j
import openai
from dotenv import load_dotenv
from pydantic import BaseModel
from neo4j_graphrag.embeddings.openai import AzureOpenAIEmbeddings as _AzureOpenAIEmbeddings
from prompts import CONTEXT_PROMPT, GITHUB_CONTEXT_PROMPT, PERSON_NAME_PROMPT
from neo4j_graphrag.llm import AzureOpenAILLM
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — PDF to Markdown
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_markdown(pdf_path: str) -> str:
    client = openai.AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2025-01-01-preview",
    )
    doc = fitz.open(pdf_path)
    fitz.TOOLS.mupdf_warnings()  # flush non-fatal structure warnings (e.g. malformed tag tree)
    parts = []
    for page in doc:
        img_bytes = page.get_pixmap(dpi=200).tobytes("png")
        b64 = base64.b64encode(img_bytes).decode()
        resp = client.chat.completions.create(
            model="gpt-5.4",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": (
                        "Estrai tutto il testo da questa pagina di CV in ordine logico "
                        "(prima la colonna sinistra dall'alto in basso, poi la colonna destra). "
                        "Restituisci solo il testo estratto in formato Markdown, senza commenti aggiuntivi."
                    )},
                ],
            }],
            max_completion_tokens=4096,
        )
        parts.append(resp.choices[0].message.content or "")
    return "\n\n".join(parts)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — Markdown chunking
# ─────────────────────────────────────────────────────────────────────────────

class MarkdownSection(BaseModel):
    breadcrumb: str
    content: str
    order: int

_HEADERS_TO_SPLIT = [
    ("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")
]

def split_markdown(md: str, chunk_size: int = 1800, chunk_overlap: int = 100) -> List[MarkdownSection]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = char_splitter.split_documents(header_splitter.split_text(md))
    out = []
    for order, doc in enumerate(docs):
        headers = [doc.metadata[k] for k in ["h1", "h2", "h3", "h4", "h5", "h6"] if doc.metadata.get(k)]
        breadcrumb = " > ".join(headers) if headers else "(root)"
        content = doc.page_content.strip()
        if content:
            out.append(MarkdownSection(breadcrumb=breadcrumb, content=content, order=order))
    return out

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
    
async def contextualize_github(
    llm: AzureOpenAILLM, full_doc: str, section: MarkdownSection, person_name: str
) -> str:
    user = (
        f"<document>\n{full_doc}\n</document>\n\n"
        f"<chunk>\nSection path: {section.breadcrumb}\n\n{section.content}\n</chunk>"
    )
    prompt = GITHUB_CONTEXT_PROMPT.format(person_name=person_name)
    response = await llm.ainvoke(input=user, system_instruction=prompt)
    return response.content.strip()

def _normalize_node_names(graph) -> None:
    for node in graph.nodes:
        name = node.properties.get("name")
        if isinstance(name, str):
            node.properties["name"] = name.strip().title()

def _anchor_person(driver: neo4j.Driver, person_name: str) -> None:
    """Pre-create the canonical Person node so the resolver can unify it with extractor outputs."""
    with driver.session() as session:
        session.run(
            """
            MERGE (p:Person:__Entity__ {name: $name})
            SET p.is_canonical = true
            """,
            {"name": person_name.strip().title()},
        )