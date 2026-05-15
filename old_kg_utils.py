import base64
import os
import re
from datetime import date
from typing import List

import fitz
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from models.llm_models import (
    DocumentSummary,
    ChunkedSummary,
    GraphExtraction,
)


FILES_FOLDER = "files/Simone Bitti"

ALLOWED_NODES = [
    "Profiency",
    "Person",
    "Role",
    "Skill",
    "ProgrammingLanguage",
    "Technology",
    "Organization",
    "PersonalProject",
    "Language",
    "Concept",
    "Contact",
    "Certification",
    "Activity",
    "Project",
    "ProjectUrl",
    "DateRange",
]

_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REL_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _sanitize_label(label: str, allowed: List[str]) -> str:
    """Return label only if present in the whitelist, else fallback to 'Entity'."""
    if label in allowed and _LABEL_RE.match(label):
        return label
    return "Entity"


def _sanitize_rel_type(rel_type: str) -> str:
    """Normalize relationship type to UPPER_SNAKE_CASE and validate against a regex."""
    candidate = re.sub(r"[^A-Za-z0-9_]+", "_", rel_type or "").upper().strip("_")
    if not candidate or not _REL_TYPE_RE.match(candidate):
        return "RELATED_TO"
    return candidate


def summarize_document(html_pages):
    llm = AzureChatOpenAI(
        azure_deployment="gpt-5.4",
        openai_api_version="2024-12-01-preview",
        temperature=0.0
    )

    text=str(html_pages)

    system_message = """
You are an expert at analyzing and summarizing HTML documents extracted from PDF pages, specifically CVs (resumes) and Cover Letters.

## Goal
Your task is to read the full text of an HTML document—extracted from a PDF page that may represent a CV or a Cover Letter—and produce:
- A brief summary (2-3 sentences) that concisely describes the overall content and purpose of the document.
- An extremely detailed summary of the person described in the document. This summary must capture all relevant information, structure, and content from the source, with a strong focus on faithfully describing the individual, their background, experience, skills, and any other available details.

## Instructions

- Carefully analyze the entire HTML document, including all sections, headings, paragraphs, lists, and any other elements, recognizing the context of CVs and Cover Letters.
- For the brief summary, use only 2-3 sentences to capture the essence and intent of the document.
- For the detailed summary, be extremely thorough and focus on the person to whom the document refers. Do not omit important points, key examples, or supporting information about the individual, their achievements, career, education, skills, personality, or any other personal attribute or context found in the document.
- Use neutral, precise language. Do not rewrite, invent, or alter facts. Do not translate or interpret the content; just summarize faithfully.
- Focus on accuracy: if any part is ambiguous or unclear, include it in the summary as best as possible without making assumptions or fabrications.
- Do not omit any significant information or section.

## Output Format

Return your response in the following JSON format:
{
  "brief_overview": "<2-3 sentence summary of the document>",
  "detailed_summary": "<extremely detailed summary of all information about the person present in the document>"
}

The output must be only valid JSON, with no extra text, explanation, or commentary.
"""

    human_message = f"Here is the HTML document to summarize: {text}"

    summary = llm.with_structured_output(DocumentSummary).invoke([SystemMessage(content=system_message), HumanMessage(content=human_message)])

    return  summary

def convert_img_to_html(images):

    system_prompt = """You are an expert in analyzing images of PDF pages and converting their content into HTML format.

## Goal
You will be provided with a set of images, each representing a page from a PDF document. The images may contain text, images, tables, and other relevant information. Your task is to carefully analyze each image and extract all available information, generating a single, faithful HTML document that combines the content and structure from all pages.

## Instructions

- Carefully examine every element in every image, including text, headings, tables, images, and other graphical or structural components.
- Extract all visible and meaningful information from each image, and combine all information into a single HTML document that reflects the full content and structure of the original PDF.
- The combined HTML must preserve the structure and hierarchy as visually represented in the original PDF, maintaining the correct order and importance of information (e.g., headers, sections, lists, tables, images), as it appears page by page.
- Do not add colors, styles, custom classes, or any CSS. Only use plain HTML to represent the extracted information and structure.
- If any image is empty or does not contain any useful or readable information, ignore that image.
- The output should be only the complete, combined HTML document, with no extra explanation or commentary.

## Output Format

- Your response must contain only a single, valid HTML document that includes all the information and structure extracted from all the provided images, in the correct order.
- Do not include any explanations, comments, or content outside of the HTML.
"""


    messages = [SystemMessage(content=system_prompt)]

    for image in images:
        messages.append(HumanMessage(content=[
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                    }
                ])
        )

    prompt = ChatPromptTemplate.from_messages(messages)

    llm = AzureChatOpenAI(
        azure_deployment="gpt-5.4",
        openai_api_version="2024-12-01-preview",
        temperature=0.0
    )

    chain = prompt | llm

    response = chain.invoke({})

    html = response.content

    return html

def get_summary_chunks(summary):
    llm = AzureChatOpenAI(
        azure_deployment="gpt-5.4",
        openai_api_version="2024-12-01-preview",
        temperature=0.0
    )

    chunking_prompt = """
You are an expert in document analysis and information structuring.

## Goal
Your task is to take a detailed summary of a CV or Cover Letter, and divide it into semantically meaningful chunks. Each chunk should represent a coherent topic, section, or set of related information. The objective is to ensure that each chunk is both self-contained and contextually meaningful, allowing for easy downstream processing and understanding.

## Instructions

- Analyze the provided summary and identify natural boundaries between sections or topics (e.g., "Education", "Work Experience", "Skills", "Certifications", "Personal Projects", etc.).
- Split the summary into chunks based on these boundaries. Avoid dividing information mid-sentence or mid-topic.
- Ensure that each chunk is long enough to be meaningful (preferably at least 3-4 sentences) but not excessively long (generally no more than 300 words per chunk).
- Retain all important context needed for each chunk to be understood independently, but do not repeat large blocks of text.
- If a section is particularly large, you may divide it into multiple logically connected chunks (e.g., "Work Experience (2018-2022)", "Work Experience (2023-present)").
- Maintain the original order of sections and information as they appeared in the summary.
- Do not omit any information or merge unrelated topics into a single chunk.
- For each chunk, identify exactly 5 keywords that best represent the content and main topic of the chunk (e.g., "CONTACT", "WORK EXPERIENCE", "EDUCATION", etc.).

## Output Format

Return your response in the following JSON format:

{
  "chunks": [
    {
      "title": "<short descriptive title for this chunk>",
      "content": "<text of this chunk>",
      "keywords": ["<KEYWORD1>", "<KEYWORD2>", "<KEYWORD3>"]
    },
    ...
  ]
}

- Each chunk must have a concise, descriptive title.
- The keywords field must be a list of exactly 3 uppercase keywords that summarize the content of the chunk.
- The output should be only valid JSON, with no extra text, explanation, or commentary.
"""



    human_message = f"Here the summary to analyze: {summary}"

    result = llm.with_structured_output(ChunkedSummary).invoke([SystemMessage(content=chunking_prompt), HumanMessage(content=human_message)])

    return  result.chunks


def extract_graph_from_chunk(llm, text: str, allowed_nodes: List[str], additional_instructions: str) -> GraphExtraction:
    """Extract entities and relationships from a single text chunk using the LLM."""
    system_prompt = f"""
You are an expert in information extraction and knowledge graph construction.

## Goal
Read the provided text and extract a knowledge graph as a list of nodes and relationships.

## Allowed node types
{", ".join(allowed_nodes)}

## Instructions
- Only emit nodes whose `type` is one of the allowed node types above. Do NOT invent new types.
- Each node MUST have a canonical `id` (the entity name, normalized: trimmed, no surrounding quotes, consistent capitalization across mentions). Reuse the same `id` for the same real-world entity so duplicates can be merged.
- Relationship `type` MUST be UPPER_SNAKE_CASE (e.g. WORKS_AT, HAS_SKILL, STUDIED_AT). Use concise, generic predicates.
- Each relationship's `source_type` and `target_type` MUST match the `type` of an emitted node, and `source_id`/`target_id` MUST match its `id`.
- Properties are optional: only add them when the value is explicitly present in the text and adds meaningful information.
- Do not fabricate facts not present in the text.

## Additional context
{additional_instructions}
"""
    human_message = f"Text to extract from:\n{text}"
    return llm.with_structured_output(GraphExtraction, method="function_calling").invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_message)]
    )


def _clear_graph_tx(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def _insert_chunk_tx(tx, doc_id: str, doc_props: dict, nodes: list, relationships: list):
    """Single transaction: upsert Document, entity nodes, relationships, MENTIONED_IN edges."""
    tx.run(
        """
        MERGE (d:Document {id: $doc_id})
        SET d += $props
        """,
        doc_id=doc_id,
        props=doc_props,
    )

    for n in nodes:
        label = _sanitize_label(n["type"], ALLOWED_NODES)
        tx.run(
            f"""
            MERGE (n:`{label}` {{id: $id}})
            SET n += $props
            WITH n
            MATCH (d:Document {{id: $doc_id}})
            MERGE (n)-[:MENTIONED_IN]->(d)
            """,
            id=n["id"],
            props=n.get("properties") or {},
            doc_id=doc_id,
        )

    for r in relationships:
        src_label = _sanitize_label(r["source_type"], ALLOWED_NODES)
        tgt_label = _sanitize_label(r["target_type"], ALLOWED_NODES)
        rel_type = _sanitize_rel_type(r["type"])
        tx.run(
            f"""
            MERGE (s:`{src_label}` {{id: $src_id}})
            MERGE (t:`{tgt_label}` {{id: $tgt_id}})
            MERGE (s)-[rel:`{rel_type}`]->(t)
            SET rel += $props
            """,
            src_id=r["source_id"],
            tgt_id=r["target_id"],
            props=r.get("properties") or {},
        )


def create_kg():
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    with driver.session() as session:
        session.execute_write(_clear_graph_tx)
        print("Graph cleared.")

        embeddings_3_large = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-large",
            openai_api_version="2024-12-01-preview",
            dimensions=3072,
        )

        llm = AzureChatOpenAI(
            azure_deployment="gpt-5.4",
            openai_api_version="2024-12-01-preview",
            temperature=0.0,
        )

        for filename in os.listdir(FILES_FOLDER):
            if not filename.lower().endswith(".pdf"):
                continue

            filepath = os.path.join(FILES_FOLDER, filename)
            doc = fitz.open(filepath)
            print(f"Processo: {filename}")

            images = []
            for page_number in range(len(doc)):
                page = doc[page_number]
                pix = page.get_pixmap()
                img_bytes = pix.tobytes(output="png")
                print(f"  Pagina {page_number+1}: {len(img_bytes)} bytes")
                images.append(base64.b64encode(img_bytes).decode("ascii"))

            html = convert_img_to_html(images)
            if not html:
                continue

            cleaned_html = html.replace("```html", "").replace("\n", "").replace("```", "")
            summary = summarize_document(cleaned_html)
            chunks = get_summary_chunks(summary.detailed_summary)

            today = date.today().strftime("%Y-%m-%d")
            additional_instructions = f"""
All the HTML documents provided belong to the same original document (e.g., a multi-page CV or Cover Letter for a single person).
Do not treat them as separate entities. Instead, merge and analyze the content as a single document, preserving the correct order and overall structure.
Use the following brief overview as context to better understand the purpose and content:
Summary of the document: {summary.brief_overview}

Today date is: {today}
"""

            for idx, chunk in enumerate(chunks):
                chunk_text = f"{chunk.title} \n {chunk.content}"
                embedding = embeddings_3_large.embed_query(chunk.content)

                extraction = extract_graph_from_chunk(
                    llm=llm,
                    text=chunk_text,
                    allowed_nodes=ALLOWED_NODES,
                    additional_instructions=additional_instructions,
                )

                print(f"  Chunk {idx+1}/{len(chunks)} '{chunk.title}': "
                      f"{len(extraction.nodes)} nodes, {len(extraction.relationships)} relationships")

                doc_id = f"{filename}::{idx}"
                doc_props = {
                    "text": chunk_text,
                    "source": filename,
                    "chunk_title": chunk.title,
                    "keywords": chunk.keywords,
                    "embedding": embedding,
                }

                session.execute_write(
                    _insert_chunk_tx,
                    doc_id,
                    doc_props,
                    [n.model_dump() for n in extraction.nodes],
                    [r.model_dump() for r in extraction.relationships],
                )

    driver.close()


if __name__ == "__main__":
    create_kg()
