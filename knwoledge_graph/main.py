"""
Ingestion pipeline:
  Source (PDF / GitHub README)
  -> Load & Chunk (header-aware, paragraph fallback)
  -> Contextual chunking (LLM situates each chunk)
  -> Embed
  -> STRICT entity/relation extraction (neo4j-graphrag v1.15+)
  -> Neo4j persistence
  -> Entity resolution

Run:
    python ingestion.py
"""
import asyncio
import os
import neo4j
from dotenv import load_dotenv
from executors import KnwoledgeGraphPipelineExecutor
from helpers import AzureOpenAIEmbeddings
from loaders import GitHubLoader, PDFLoader
from neo4j_graphrag.llm import AzureOpenAILLM

load_dotenv()

FILES_FOLDER = "files"

async def run_ingestion() -> None:
    driver = neo4j.GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    llm = AzureOpenAILLM(
        model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        model_params={"temperature": 0.0},
    )
    
    embedder = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        dimensions=3072,
    )
  
    pdf_loader = PDFLoader(FILES_FOLDER, llm, driver)
    github_loader = GitHubLoader(
        llm=llm,
        person_name=pdf_loader.person_name,
        github_token=os.getenv("GITHUB_TOKEN"),
    )

    loaders = [
        pdf_loader,
        github_loader,
    ]

    pipeline = KnwoledgeGraphPipelineExecutor(driver, llm, embedder, loaders=loaders)

    await pipeline.run()

    driver.close()

    print("\nIngestion complete.")

if __name__ == "__main__":
    asyncio.run(run_ingestion())
