from neo4j_graphrag.experimental.components.resolver import (
    BasePropertySimilarityResolver,
)
from neo4j_graphrag.experimental.components.types import ResolutionStats
from neo4j_graphrag.embeddings import AzureOpenAIEmbeddings
import numpy as np

class AzureEmbeddingResolver(BasePropertySimilarityResolver):
    def __init__(
        self,
        driver,
        embedder: AzureOpenAIEmbeddings,
        filter_query=None,
        resolve_properties=None,
        similarity_threshold: float = 0.92,
        neo4j_database=None,
    ):
        super().__init__(
            driver,
            filter_query=filter_query,
            resolve_properties=resolve_properties or ["name", "description"],
            similarity_threshold=similarity_threshold,
            neo4j_database=neo4j_database,
        )
        self.embedder = embedder
        self._cache: dict[str, list[float]] = {}

    async def run(self) -> ResolutionStats:
        return await super().run()

    def _embed(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = self.embedder.embed_query(text)
        return self._cache[text]

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        a = np.array(self._embed(text_a))
        b = np.array(self._embed(text_b))
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))