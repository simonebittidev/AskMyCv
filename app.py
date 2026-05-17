import json
from typing import List, Optional
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pathlib import Path
from langchain_neo4j import Neo4jGraph
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import neo4j
from langchain_core.runnables import RunnableLambda
from models.llm_models import RewrittenQuestion
from retriever.custom_retriever import CustomKnowledgeGraphRetriever
import ssl
from langchain_core.callbacks.base import BaseCallbackHandler
from datetime import date
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import json
from sentence_transformers import CrossEncoder
from prompts import rewrite_question_prompt, final_answer_prompt

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
    with neo4j_graph._driver.session() as session:
        session.run("RETURN 1")
    with _driver.session() as session:
        session.run("RETURN 1")
    print("Keeping Neo4j Aura instance alive...")

scheduler.add_job(keep_neo4j_alive, "interval", minutes=20, next_run_time=datetime.now())
scheduler.start()

# Serve React app (build)
client_path = Path("client/out")
app.mount("/static", StaticFiles(directory=client_path), name="static")

_driver = neo4j.GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        max_connection_lifetime=180,
    )

_embedder = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        dimensions=int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", 3072)),
    )

_reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

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
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.7 #more creative and less deterministic responses
    )

    messages = [SystemMessage(content=rewrite_question_prompt), HumanMessage(content=f"User's input: {state['messages'][-1]}\nChat history: {state['history']}")]

    result = llm_history.with_structured_output(RewrittenQuestion).invoke(messages)
    rewritten_question = result.rewritten_question
    
    return {"rewritten_question": rewritten_question}

@app.post("/")
async def health_check_post():
    return {"status": "ok"}

@app.get("/download-cv")
async def download_cv():
    file_path = Path("files/simonebitticv.pdf")
    if file_path.exists():
        return FileResponse(path=file_path, filename="simonebitticv.pdf", media_type="application/pdf")
    raise HTTPException(status_code=404, detail="CV file not found")

def generate_final_answer(state: State):
    message = state["rewritten_question"]
    context = state["context"]
    today = state.get("today", date.today().isoformat())

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.7 #more creative and less deterministic responses
    )

    result = llm.invoke([SystemMessage(content=final_answer_prompt.format(message=message, context=context, today=today))])
    return {"messages": result}

def send_end(state: State):
    return {"messages": AIMessage(content="[DONE]")}

def search_documents(state: State):
    query = state["rewritten_question"] or state["messages"][-1].content

    retriever = CustomKnowledgeGraphRetriever(
        driver=_driver,
        embedder=_embedder,
        reranker=_reranker
    )

    context = retriever.search(query)

    return {"context": context}

@app.get("/stream")
async def stream_sse(text: str, history: str):
    async def event_generator(text, history):
        try:
            print(f"Received text: {text}")
            print(f"Received history: {history}")
            
            graph_builder = StateGraph(State)
            graph_builder.add_node("rewrite_question", RunnableLambda(rewrite_question).with_config(tags=["nostream"]))
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
