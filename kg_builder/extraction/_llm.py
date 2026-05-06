"""Shared LLM / embeddings factories — single place to upgrade models."""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from kg_builder.config import (
    EMBEDDING_DEPLOYMENT,
    EMBEDDING_DIMENSIONS,
    LLM_DEPLOYMENT,
    OPENAI_API_VERSION,
)


@lru_cache(maxsize=4)
def get_chat_llm(temperature: float = 0.0) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=LLM_DEPLOYMENT,
        openai_api_version=OPENAI_API_VERSION,
        temperature=temperature,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_DEPLOYMENT,
        openai_api_version=OPENAI_API_VERSION,
        dimensions=EMBEDDING_DIMENSIONS,
    )
