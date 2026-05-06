import json
from typing import List, Optional
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from models.llm_models import RewrittenQuestion
from utils import grade_document, get_stractered_data, get_context
from kg_builder.retrieval import build_retriever
from kg_builder.config import LLM_DEPLOYMENT, OPENAI_API_VERSION
import ssl
from langchain.callbacks.base import BaseCallbackHandler
from datetime import date
import time
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

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
        azure_deployment=LLM_DEPLOYMENT,
        openai_api_version=OPENAI_API_VERSION,
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
        azure_deployment=LLM_DEPLOYMENT,
        openai_api_version=OPENAI_API_VERSION,
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
    """Hybrid retrieval: vector + project-aware boost + community summaries."""
    retriever = build_retriever()
    chunks = retriever.retrieve(state["rewritten_question"])
    return {"unstructered_data": [c.to_string() for c in chunks]}

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
        azure_deployment=LLM_DEPLOYMENT,
        openai_api_version=OPENAI_API_VERSION,
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
    
@app.get("/stream")
async def stream_sse(text: str, history: str):
    async def event_generator(text, history):
        try:
            print(f"Received text: {text}")
            print(f"Received history: {history}")
            
            graph_builder = StateGraph(State)
            graph_builder.add_node("rewrite_question", RunnableLambda(rewrite_question).with_config(tags=["nostream"]))
            graph_builder.add_node("get_structered_data", RunnableLambda(get_structered_data).with_config(tags=["nostream"]))
            graph_builder.add_node("get_unstructered_data", RunnableLambda(get_unstructered_data).with_config(tags=["nostream"]))
            graph_builder.add_node("grade_documents_and_get_context", RunnableLambda(grade_documents_and_get_context).with_config(tags=["nostream"]))
            graph_builder.add_node("generate_final_answer", generate_final_answer)
            graph_builder.add_node("send_end", send_end)

            graph_builder.add_edge(START, "rewrite_question")
            graph_builder.add_edge("rewrite_question", "get_structered_data")
            graph_builder.add_edge("rewrite_question", "get_unstructered_data")
            graph_builder.add_edge("get_unstructered_data", "grade_documents_and_get_context")
            graph_builder.add_edge("get_structered_data", "grade_documents_and_get_context")
            graph_builder.add_edge("grade_documents_and_get_context", "generate_final_answer")
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
