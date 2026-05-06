# ChatMyCv

A personal study project that turns your CV and cover letter into an interactive knowledge graph you can query via a chat interface.

---

## 🚀 Overview

**ChatMyCv** is a chat-bot application that lets you explore my professional background, skills and projects by asking natural-language questions. Under the hood:

1. **Multi-source ingestion** powered by the `kg_builder` package:
   - CV and cover letter PDFs are parsed to **Markdown** with [Docling](https://github.com/DS4SD/docling) — no vision-LLM needed for the parsing step.
   - **Personal projects come from GitHub READMEs** (curated list in `config/projects.yml`), fetched directly via the GitHub API. The cover letter is no longer used to extract projects.
2. **Per-source schemas** keep the graph clean: the cover letter cannot create `PersonalProject` nodes, those are anchored deterministically from GitHub metadata.
3. **Storage** in a Neo4j knowledge graph with vector-bearing `Document` chunks (late-chunking blended embeddings) plus `CommunitySummary` nodes for GraphRAG-style global search.
4. **Hybrid retrieval** at query time: vector + fulltext + graph traversal + project boost + community summaries.
5. **Querying**: user questions are translated into Cypher by an LLM and merged with the hybrid retrieval context.
6. **Real-time streaming** of answers via Server-Sent Events (SSE), orchestrated with LangGraph.

This project is purely experimental, a playground for combining NLP, graph databases and modern backend/frontend tooling.

---

## 🔧 Tech Stack

- **Backend**:  
  - Python 3.x, [FastAPI](https://fastapi.tiangolo.com/)  
  - Neo4j Aura (cloud-hosted)  
  - APScheduler (to keep the Aura instance “warm”)  
  - LangGraph (flow orchestration)  
  - Azure OpenAI GPT for NLP tasks

- **Frontend**:  
  - Next.js / React / TypeScript  
  - Server-Sent Events for live response streaming  
  - Tailwind CSS & Heroicons  

---

## 📦 Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/ChatMyCv.git
cd ChatMyCv
```

### 2. Backend setup
  1.	Create a virtual environment and install dependencies:

  ```bash
  python -m venv .venv
  source .venv/bin/activate      # macOS/Linux
  .venv\Scripts\activate         # Windows
  pip install -r requirements.txt
  ```

  2.	Set environment variables in a `.env` file (see `.env.example`):

  ```dotenv
  AZURE_OPENAI_ENDPOINT=<your-azure-openai-endpoint>
  AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>
  NEO4J_URI=<your-neo4j-aura-uri>
  NEO4J_USERNAME=<username>
  NEO4J_PASSWORD=<password>
  # Required for GitHub README ingestion (5000 req/h vs 60)
  GITHUB_TOKEN=<your-personal-access-token>
  GITHUB_USERNAME=simonebitti
  ```

  3.	Curate the project list in `config/projects.yml` (each entry: `repo: "owner/name"`).

  4.	Build the knowledge graph:

  ```bash
  python -m kg_builder
  ```

  5.	Start the FastAPI server:

  ```bash
  uvicorn app:app --reload
  ```

### 3. Frontend setup

```bash
cd client
npm install
npm run dev
```

## ⚙️ Usage
	1.	Open your browser at http://localhost:3000.
	2.	You’ll see suggested questions on first load (e.g. “What technologies does Simone have experience with?”).
	3.	Type or click a suggestion—responses will stream back in real time.
	4.	Ask anything about my CV, cover letter or projects!

## 🛠 Pipeline Details

The `kg_builder/` package owns the entire ingestion pipeline. Each module has
a single responsibility and can be reused or replaced independently:

```
kg_builder/
├── pipeline.py          # orchestrator: run_pipeline()
├── config.py            # Settings + projects.yml loader
├── models.py            # Pydantic models (SourceDocument, ProjectDoc, ...)
├── sources/             # PdfSource (Docling) + GitHubSource (httpx)
├── extraction/          # summarizer, chunker (late chunking), graph_extractor
├── graph/               # Neo4j client, per-source schema, loader, communities
└── retrieval/           # hybrid retriever used by app.py at query time
```

Pipeline stages (`python -m kg_builder`):

1. **Wipe** the existing Neo4j graph and `MERGE` the `Person` anchor.
2. **PDFs → Markdown** via Docling (fallback: vision-LLM Markdown).
3. **GitHub READMEs** fetched for every repo in `config/projects.yml`.
4. **Summarise → semantic chunk → embed** with late-chunking blend
   (chunk vector mixed with the document-level vector for global context).
5. **Anchor `PersonalProject` nodes** deterministically from GitHub metadata
   (`MERGE` on repo full name) — the LLM never invents projects.
6. **LLM graph extraction** under per-source schemas: CV / cover letter is
   forbidden from producing `PersonalProject`; READMEs are restricted to
   `Technology`, `ProgrammingLanguage`, `Concept`, `Topic`, `Skill`.
7. **Community detection** (Leiden via Neo4j GDS, with `leidenalg` fallback)
   produces `CommunitySummary` nodes for GraphRAG-style global queries.

At query time, `app.py` uses `kg_builder.retrieval.HybridGraphRetriever`,
which combines vector + fulltext on `Document` chunks, project-aware
context boost, and community summaries.

⸻

## 🤝 Contributing

This is a solo study project, but I’d love to hear your ideas!
	•	Report issues or suggest features via GitHub Issues.
	•	Feel free to submit pull requests if you improve extraction prompts, graph models or frontend UX.

⸻

## 📜 License

Distributed under the MIT License. See LICENSE for more details.


 
