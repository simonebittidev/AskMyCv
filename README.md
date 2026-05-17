<h1>
  <img src="https://github.com/user-attachments/assets/0b7467a4-2a0b-47e3-93fa-0dcfedad827c" 
       width="40" height="40" alt="logo" valign="middle" />
  &nbsp;AskMyCv
</h1>

A personal study project that turns a CV and GitHub profile into an interactive knowledge graph you can query via a chat interface.

---

## Overview

**AskMyCv** is a chatbot that lets you explore a professional background, skills and projects by asking natural-language questions.

The system has two independent phases:

1. **Ingestion** — documents are parsed, chunked, embedded and written into a Neo4j knowledge graph. This runs once (or whenever the source material changes).
2. **Retrieval and generation** — at query time a hybrid search retrieves the most relevant passages from the graph, a reranker selects the best ones, and an LLM streams the final answer.

The chat flow is orchestrated with LangGraph.

---

## Tech stack

**Backend**

- Python 3, FastAPI
- Neo4j Aura (cloud-hosted graph database)
- neo4j-graphrag — hybrid retrieval with custom Cypher expansion
- LangGraph — graph-based flow orchestration
- LangChain — LLM wrappers and prompt templates
- Azure OpenAI — GPT for generation, rewriting and entity extraction; `text-embedding-3-large` for embeddings
- sentence-transformers `BAAI/bge-reranker-v2-m3` — multilingual cross-encoder reranker
- APScheduler — keeps the Aura free-tier instance alive

**Frontend**

- Next.js, React, TypeScript
- Tailwind CSS
- Server-Sent Events for streaming responses

---

## Ingestion pipeline

The ingestion pipeline lives in `knwoledge_graph/` and is run once to populate the Neo4j database. It handles two source types: PDF documents (CV, cover letter) and GitHub READMEs.

### 1. Document loading and chunking

**PDFs** are rendered page by page as PNG images (200 DPI) and sent to Azure OpenAI's vision endpoint, which converts them to Markdown. This approach avoids the brittleness of text-extraction libraries and correctly handles multi-column layouts.

The resulting Markdown is split into sections with `MarkdownHeaderTextSplitter`, which preserves the document hierarchy (`#`, `##`, `###`). Sections larger than 1800 characters are further split with `RecursiveCharacterTextSplitter` (100-character overlap).

**GitHub READMEs** are fetched from the GitHub API (see `config/projects.yml` for the list of repositories). Badge images and inline media are stripped before chunking to avoid wasting tokens on non-semantic content.

### 2. Contextual enrichment

Each chunk is "situated" in its source document by a dedicated LLM call before extraction. The model receives the full document and the target section and produces 2–3 sentences describing how the section fits in the overall document. This context is stored alongside the chunk and is critical for correct entity disambiguation (e.g., the same technology name appearing in different roles is resolved differently).

GitHub chunks receive a variant of this prompt that explicitly identifies the chunk as part of a personal project by the CV subject.

### 3. Entity and relation extraction

Entities and relationships are extracted from each enriched chunk using `LLMEntityRelationExtractor` from neo4j-graphrag, operating against a strict schema defined in `schema.py`.

**PDF/CV schema** defines 24 node types (Person, Role, Organization, ProgrammingLanguage, Framework, Database, Project, Degree, Certification, …) and 21 relationship types (WORKS_AT, WORKED_AT, HAS_SKILL, STUDIED_AT, USES, DEPLOYED_ON, …), with 40 explicit triplet patterns (e.g. only `Person → WORKS_AT → Organization`, not the reverse). No types outside this schema can be created.

**GitHub schema** is a separate, lighter schema (11 node types, 13 relationship types) focused on projects: CONTRIBUTED_TO, BUILT_WITH, WRITTEN_IN, DEPENDS_ON, INTEGRATES_WITH, IMPLEMENTS, SOLVES, etc.

The use of strict schemas prevents the LLM from hallucinating node or relationship types.

Before extraction, a canonical `Person` node is pre-created in Neo4j with `is_canonical=true`. This anchor guarantees that the entity resolver has a stable merge target for the CV subject across all chunks and source documents.

### 4. Embedding and writing

Each chunk is embedded with `text-embedding-3-large` (3072 dimensions) and stored in Neo4j alongside its extracted entities and relationships. A vector index (`chunk_embedding`) and a fulltext index (`chunk_fulltext`) are created to support hybrid retrieval.

### 5. Entity resolution

After all sources are ingested, a two-phase deduplication pass merges redundant entity nodes:

- **Phase 1 — exact match:** `SinglePropertyExactMatchResolver` merges `__Entity__` nodes with identical `name` values.
- **Phase 2 — semantic similarity:** `AzureEmbeddingResolver` embeds entity names and descriptions and merges pairs with cosine similarity above 0.92. Embeddings are cached to avoid redundant API calls.

The high similarity threshold (0.92) is intentional: false merges are more damaging to retrieval quality than residual duplicates.

---

## Retrieval and generation

At query time the LangGraph pipeline runs four nodes:

1. **rewrite_question** — the user's message and chat history are passed to an LLM that rewrites them into a single self-contained query, resolving co-references and colloquialisms.
2. **search_documents** — `CustomKnowledgeGraphRetriever` (a subclass of `HybridCypherRetriever`) runs a hybrid vector + fulltext search. A custom Cypher expansion query walks the graph from each matched chunk to collect related entities, relationships and neighbouring chunks. The top-20 candidates are reranked by the cross-encoder and the best 5 are assembled into a structured context (primary passages, related passages, entities, relations).
3. **generate_final_answer** — the rewritten question and the context are injected into the system prompt and the LLM streams the answer.
4. **send_end** — a `[DONE]` sentinel is sent to signal the end of the stream.

The FastAPI `/stream` endpoint exposes this pipeline as a Server-Sent Events stream.

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/AskMyCv.git
cd AskMyCv
```

### 2. Backend

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

Start the server:

```bash
uvicorn app:app --reload
```

### 3. Frontend

```bash
cd client
npm install
npm run dev
```

Open `http://localhost:3000`.

---

## Running ingestion

Place PDF files in `files/<person name>/` and list any GitHub repositories in `config/projects.yml`:

```yaml
projects:
  - repo: your-username/your-repo
    display_name: My Project
```

Then run:

```bash
python knwoledge_graph/main.py
```

This will clean the existing graph, ingest all sources and resolve duplicate entities.

---

## Environment variables

See `.env.example` for the full list.

| Variable | Description |
|---|---|
| `NEO4J_URI` | Neo4j Aura connection URI |
| `NEO4J_USERNAME` | Neo4j username |
| `NEO4J_PASSWORD` | Neo4j password |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | API version (e.g. `2024-12-01-preview`) |
| `AZURE_OPENAI_DEPLOYMENT` | Chat model deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment name |
| `AZURE_OPENAI_EMBEDDING_DIMENSIONS` | Embedding dimensions (default: `3072`) |
| `GITHUB_TOKEN` | Personal access token for GitHub README ingestion |

---

## License

MIT — see [LICENSE](LICENSE).
