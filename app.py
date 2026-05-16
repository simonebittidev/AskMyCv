import json
from typing import List, Optional
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import neo4j
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from models.llm_models import RewrittenQuestion
from utils import grade_document, get_unstructured_data, get_stractered_data, get_context
import ssl
from langchain_core.callbacks.base import BaseCallbackHandler
from datetime import date
import time
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.types import RetrieverResultItem
import json
from sentence_transformers import CrossEncoder

class State(TypedDict):
    messages: Annotated[list, add_messages]
    history: Optional[dict]
    rewritten_question: Optional[str]
    context: Optional[str]
    data: Optional[dict]
    structered_data: Optional[dict]
    structered_data_documents: Optional[List[str]]
    unstructered_data: Optional[List[str]]
    is_end: Optional[bool]


print(ssl.OPENSSL_VERSION)

load_dotenv()

app = FastAPI()

print(f"NEO4J_URI: {os.getenv('NEO4J_URI')}")
print(f"NEO4J_PASSWORD: {os.getenv('NEO4J_PASSWORD')}")
print(f"NEO4J_USERNAME: {os.getenv('NEO4J_USERNAME')}")

neo4j_graph = Neo4jGraph(url=os.getenv('NEO4J_URI'), username=os.getenv('NEO4J_USERNAME'), password=os.getenv('NEO4J_PASSWORD'), enhanced_schema=True, driver_config={"max_connection_lifetime": 180})

# Scheduler to keep Neo4j Aura instance alive
scheduler = BackgroundScheduler()
def keep_neo4j_alive():
    # Execute a simple Cypher query to keep the connection active
    with neo4j_graph._driver.session() as session:
        session.run("RETURN 1")
        print("Keeping Neo4j Aura instance alive...")

# Schedule the job to run once every day
scheduler.add_job(keep_neo4j_alive, "interval", days=1, next_run_time=datetime.now())
scheduler.start()

# Serve React app (build)
client_path = Path("client/out")
app.mount("/static", StaticFiles(directory=client_path), name="static")

class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = []

    def on_llm_new_token(self, token: str, **kwargs):
        self.queue.append(token)

    def get(self):
        while self.queue:
            yield self.queue.pop(0)

def rewrite_question(state: State):
    llm_history = AzureChatOpenAI(
        azure_deployment="gpt-5.4",
        openai_api_version="2024-12-01-preview",
        temperature=0.7 #more creative and less deterministic responses
    )

    rewrite_question_prompt = """
You are an expert assistant in communication clarity and context analysis.

## Goal
Your task is to analyze the user's latest question together with the conversation history. If the question is unclear or ambiguous, you must use the conversation history to rewrite it in a way that makes it clearer and more precise, while preserving the original intent and all important details. If the question is already clear, simply return it as is.

## Instructions

- Carefully read the user's question and the conversation history.
- If the question is ambiguous, vague, or hard to understand, rewrite it to be as clear and precise as possible, using relevant context from the conversation history to clarify meaning.
- Do not invent or add information that is not present in the conversation.
- If the question is already clear and unambiguous, return it exactly as it was given.
- Always use the same language as the user's question.
- Do not answer the question or provide any additional information—your only task is to rewrite it if needed.

## Output Format

Return your response in the following JSON format:

{
"rewritten_question": "<the clarified version of the user's question, or the original if it was already clear>"
}

- The output must be only valid JSON, with no extra explanation or commentary.
"""

    messages = [SystemMessage(content=rewrite_question_prompt), HumanMessage(content=f"User's input: {state['messages'][-1]}\nChat history: {state['history']}")]

    result = llm_history.with_structured_output(RewrittenQuestion).invoke(messages)
    rewritten_question = result.rewritten_question
    
    return {"rewritten_question": rewritten_question}

def get_structered_data(state: State):
    llm = AzureChatOpenAI(
        azure_deployment="gpt-5.4",
        openai_api_version="2024-12-01-preview",
        temperature=0.7 #more creative and less deterministic responses
    )

    text2cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Given an input question, convert it to a Cypher query. No pre-amble. "
                "Do not wrap the response in any backticks or anything else. Respond with Cypher statements only!"
            ),
        ),
        (
            "human",
            (
                """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
Do not wrap the response in any backticks or anything else. Respond with Cypher statements only!

Here is the schema information
{schema}

User input: {question}

Return your answer in the following JSON format:
{{
  "main_query": "<a Cypher query to answer the user's question>",
  "document_distinct_query": "<a Cypher query that, using the same pattern as main_query, returns an array of strings, each string being the full text (for example, the 'text' property) of a DISTINCT Document node from which the main_query results are extracted. The output of this query must be a list of strings, each containing the full text of a relevant document.>"
}}
"""
            ),
        ),
    ]
    )

    text2cypher_chain = text2cypher_prompt | llm | StrOutputParser()

    neo4j_graph._driver.verify_connectivity()

    structured_data, documents = get_stractered_data(neo4j_graph, state["rewritten_question"], text2cypher_chain)

    return {"structered_data":structured_data, "structered_data_documents": documents}

@app.get("/download-cv")
async def download_cv():
    file_path = Path("files/Simone Bitti/Simone Bitti CV.pdf")
    if file_path.exists():
        return FileResponse(path=file_path, filename="Simone Bitti CV.pdf", media_type="application/pdf")
    raise HTTPException(status_code=404, detail="CV file not found")

def get_unstructered_data(state: State):
    embeddings_3_large : AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-large",
        openai_api_version="2024-12-01-preview",
        dimensions=3072
    )

    vector_index = Neo4jVector.from_existing_graph(
        embedding=embeddings_3_large,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

    unstructered_data = get_unstructured_data(vector_index, state["rewritten_question"])

    return {"unstructered_data":unstructered_data}

def generate_final_answer(state: State):
    message = state["rewritten_question"]
    context = state["context"]
    today = state.get("today", date.today().isoformat())

    template = f"""
You are Ask my cv, a virtual assistant designed to answer user questions about Simone. Your purpose is to provide precise, well-structured answers that help the user understand Simone's professional profile, skills, and experiences.

## Provided Information
You will receive:
- The user's question.
- Context information including:
    1. Structured data from a knowledge graph.
    2. Unstructured data from documents in a vector database, all relevant to Simone.

## Goal
Analyze all the provided context and craft a final response that highlights Simone's strengths and relevant experiences, always staying grounded in the information available.

## Instructions

- Use **only** information provided in the context.
- Do **not** mention your sources or explain how you derived your answer. Respond as if you already know Simone.
- Structure your response clearly:
    - Begin with a concise, direct answer or summary.
    - Use paragraphs to organize separate ideas.
    - Highlight important skills, achievements, or traits in **bold** (Markdown).
    - Use **Markdown formatting only**.
    - Use bullet points (`-`) for lists.
    - Insert a horizontal line (`---`) when needed for clarity.
    - Format links as [text](url).
- Rephrase and summarize the context. Do **not** copy-paste or invent information.
- Never exaggerate Simone’s responsibilities or achievements. Only mention leadership, management, or specific results if explicitly stated in the context. Do not attribute to Simone any role or accomplishment not present in the context.
- The tone must be natural and friendly, helping the user quickly understand Simone's background and skills.
- Make the answer polished and easy to appreciate.
- Never mention the context, your assistant role, or explain the type of answer you are giving. Just provide the answer directly.

## Special Instructions

- For casual, playful, or off-topic (chitchat) questions, reply with a witty, ironic tone, always in support of Simone, but without exaggeration. Be relatable, add humor, but remain credible.
- If asked about your identity, say you are Ask my cv, a virtual assistant designed to answer questions about Simone.
- If the user asks to download Simone's CV, you can provide the following answer "[Scarica il CV di Simone](/download-cv)".

## Output Language
Always respond in the **same language** as the user's input.

User Input: {message}

Context: {context}

Today's date: {today}
"""

    llm = AzureChatOpenAI(
        azure_deployment="gpt-5.4",
        openai_api_version="2024-12-01-preview",
        temperature=0.7 #more creative and less deterministic responses
    )

    result = llm.invoke([SystemMessage(content=template)])
    return {"messages": result}

def grade_documents_and_get_context(state: State):
    documents = state.get("structered_data_documents", []) + state.get("unstructered_data", [])
    data = grade_document(
        question=state["rewritten_question"],
        documents=documents)
    
    context = get_context(state["structered_data"], data)
    # context = get_context(state["structered_data"], documents)

    return {"context": context}

def send_end(state: State):
    return {"messages": AIMessage(content="[DONE]")}

def format_record(record):
    """Mappa un Record Neo4j in RetrieverResultItem strutturato."""
    seed_text = record.get("seed_text") or ""
    expanded_texts = record.get("expanded_texts") or []
    entity_facts = record.get("entity_facts") or []
    relation_facts = record.get("relation_facts") or []
    chunk_id = record.get("chunk_id") or ""

    # content: stringa human-readable (utile per debug e fallback)
    content_parts = []
    if seed_text:
        content_parts.append(f"SEED: {seed_text}")
    if expanded_texts:
        content_parts.append("EXPANDED:\n" + "\n".join(f"- {t}" for t in expanded_texts))
    if entity_facts:
        content_parts.append("ENTITIES:\n" + "\n".join(f"- {e}" for e in entity_facts))
    if relation_facts:
        content_parts.append("RELATIONS:\n" + "\n".join(f"- {r}" for r in relation_facts))
    content = "\n\n".join(content_parts)

    return RetrieverResultItem(
        content=content,
        metadata={
            "chunk_id": chunk_id,
            "seed_text": seed_text,
            "expanded_texts": expanded_texts,
            "entity_facts": entity_facts,
            "relation_facts": relation_facts,
        },
    )

def build_context(retriever_result, max_chars: int = 30000):
    """Assembla un context strutturato e deduplicato dai risultati del retriever."""
    seed_blocks = []
    expanded_set = set()       # dedup globale dei chunk espansi
    entities_set = set()
    relations_set = set()

    seen_chunk_ids = set()

    items = retriever_result.items if hasattr(retriever_result, "items") else retriever_result
    for item in items:
        meta = item.metadata or {}
        chunk_id = meta.get("chunk_id", "")
        seed_text = (meta.get("seed_text") or "").strip()

        if seed_text and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            seed_blocks.append(f"[chunk:{chunk_id}]\n{seed_text}")

        for txt in meta.get("expanded_texts", []):
            if txt and txt.strip():
                expanded_set.add(txt.strip())

        for ent in meta.get("entity_facts", []):
            if ent and ent.strip():
                entities_set.add(ent.strip())

        for rel in meta.get("relation_facts", []):
            if rel and rel.strip():
                relations_set.add(rel.strip())

    # rimuovi gli expanded che sono già nei seed (overlap testuale)
    expanded_set = {e for e in expanded_set if e not in {b.split('\n', 1)[1] for b in seed_blocks if '\n' in b}}

    parts = []
    if seed_blocks:
        parts.append("## Primary passages\n" + "\n\n".join(seed_blocks))
    if expanded_set:
        parts.append(
            "## Related passages\n" 
            + "\n\n".join(f"- {e}" for e in sorted(expanded_set))
        )
    if entities_set:
        parts.append(
            "## Entities\n" 
            + "\n".join(f"- {e}" for e in sorted(entities_set))
        )
    if relations_set:
        parts.append(
            "## Relations\n" 
            + "\n".join(f"- {r}" for r in sorted(relations_set))
        )

    context = "\n\n".join(parts)

    # hard cap sul context (troncamento "dal fondo" per preservare le primary passages)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[... context truncated ...]"

    return context


_reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")  # multilingue, eccellente

def rerank_items(query, items, docs=None, top_n=5):
    if not items:
        return []
    pairs = [(query, doc) for doc in (docs or [item.content for item in items])]
    scores = _reranker.predict(pairs)
    scored = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)
    return [
        (lambda it, s: (it.metadata.__setitem__("rerank_score", float(s)), it)[1])(it, s)
        for it, s in scored[:top_n]
    ]

def search_documents(state: State):
    driver = neo4j.GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    embedder = AzureOpenAIEmbeddings(
        model="text-embedding-3-large",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="text-embedding-3-large",
        api_version="2024-12-01-preview",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        dimensions=3072,
    )

    retrieval_query = """
MATCH (node)<-[:FROM_CHUNK]-(entity:__Entity__)
OPTIONAL MATCH (entity)-[rel]-(related:__Entity__)
WHERE NOT type(rel) IN ['FROM_CHUNK', 'NEXT_CHUNK', 'PART_OF', 'CHILD_OF']
OPTIONAL MATCH (related)-[:FROM_CHUNK]->(related_chunk:Chunk)
  WHERE related_chunk <> node
WITH 
  node,
  collect(DISTINCT entity)[..15]          AS entities,
  collect(DISTINCT rel)[..20]             AS relations,
  collect(DISTINCT related_chunk)[..5]    AS expanded_chunks
RETURN 
  elementId(node)                                                     AS chunk_id,
  node.text                                                           AS seed_text,
  [c IN expanded_chunks | c.text]                                     AS expanded_texts,
  [e IN entities | e.name + ': ' + coalesce(e.description, '')]       AS entity_facts,
  [r IN relations 
    | startNode(r).name + ' --' + type(r) + '--> ' + endNode(r).name] AS relation_facts
"""

    retriever = HybridCypherRetriever(
        driver=driver,
        vector_index_name="chunk_embedding",
        fulltext_index_name="chunk_fulltext",
        embedder=embedder,
        retrieval_query=retrieval_query,
        result_formatter=format_record,    # ← AGGIUNGI QUESTO
    )

    query = state["rewritten_question"] or state["messages"][-1].content
    results = retriever.search(query_text=query, top_k=20)

    docs_for_rerank = [
        item.metadata.get("seed_text", item.content)[:2000] 
        for item in results.items
    ]

    top_items = rerank_items(query, results.items, docs=docs_for_rerank, top_n=5)

    context = build_context(top_items)

    print("Generated context:\n", results)
    driver.close()

    return {"context": context}
   


@app.get("/stream")
async def stream_sse(text: str, history: str):
    async def event_generator(text, history):
        try:
            print(f"Received text: {text}")
            print(f"Received history: {history}")
            
            graph_builder = StateGraph(State)
            graph_builder.add_node("rewrite_question", RunnableLambda(rewrite_question).with_config(tags=["nostream"]))
            # graph_builder.add_node("get_structered_data", RunnableLambda(get_structered_data).with_config(tags=["nostream"]))
            # graph_builder.add_node("get_unstructered_data", RunnableLambda(get_unstructered_data).with_config(tags=["nostream"]))
            # graph_builder.add_node("grade_documents_and_get_context", RunnableLambda(grade_documents_and_get_context).with_config(tags=["nostream"]))
            graph_builder.add_node("search_documents", RunnableLambda(search_documents).with_config(tags=["nostream"]))
            graph_builder.add_node("generate_final_answer", generate_final_answer)
            graph_builder.add_node("send_end", send_end)

            graph_builder.add_edge(START, "rewrite_question")
            graph_builder.add_edge("rewrite_question", "search_documents")
            graph_builder.add_edge("search_documents", "generate_final_answer")
            graph_builder.add_edge("generate_final_answer", "send_end")
            graph_builder.add_edge("send_end", END)

            graph = graph_builder.compile()
            
            message=text
            history = json.loads(history)
            
            print(f"Received message: {message}")
            print(f"uri {os.getenv('NEO4J_URI')}")
            
            async for state in graph.astream({"messages": [{"role": "user", "content": message}], "history": history}, stream_mode="messages"):
                print("STATE STREAMED:", state)

                token = state[0].content
                if token:
                    if token == "[DONE]":
                        yield "data: [DONE]\n\n"
                    else:
                        yield f"data: {json.dumps({'role': 'ai', 'content': token})}\n\n"
        
        except Exception as e:
            print(f"Error in event generator: {e}")
            yield f"data: {json.dumps({'role': 'ai', 'content': '[ERROR]'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(text, history), media_type="text/event-stream")

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    target_path = client_path / full_path
    if target_path.is_dir():
        index = target_path / "index.html"
        if index.exists():
            return FileResponse(index)
    elif target_path.exists():
        return FileResponse(target_path)

    fallback_index = client_path / "index.html"
    return FileResponse(fallback_index)
