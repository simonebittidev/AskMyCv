
from typing import Optional
import neo4j
from helpers import AzureOpenAIEmbeddings, _normalize_node_names
from loaders import SourceLoader
from resolvers import AzureEmbeddingResolver
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.resolver import (
    SinglePropertyExactMatchResolver,
)
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
)
from neo4j_graphrag.experimental.components.types import (
    LexicalGraphConfig,
)

class KnwoledgeGraphPipelineExecutor:
    def __init__(
        self,
        driver: neo4j.Driver,
        llm: AzureOpenAILLM,
        embedder: AzureOpenAIEmbeddings,
        loaders: Optional[list[SourceLoader]] = None,
    ) -> None:
        self._driver = driver
        self._embedder_component = TextChunkEmbedder(embedder=embedder)
        self._raw_embedder = embedder
        self._extractor = LLMEntityRelationExtractor(
            llm=llm,
            on_error=OnError.IGNORE,
            create_lexical_graph=True,
            use_structured_output=False,
        )
        self._writer = Neo4jWriter(driver=driver)
        self._lexical_cfg = LexicalGraphConfig()
        self._loaders = loaders or []


    async def ingest(self, loader: SourceLoader) -> None:
        """Run one loader through the embed → extract → write stages."""

        schema = await SchemaBuilder().run(
            node_types=loader.node_types,
            relationship_types=loader.relationship_types,
            patterns=loader.patterns,
            additional_node_types=False,
            additional_relationship_types=False,
            additional_patterns=False,
        )
        async for batch in loader.iter_batches():
            embedded = await self._embedder_component.run(text_chunks=batch.chunks)
            graph = await self._extractor.run(
                chunks=embedded,
                document_info=batch.document_info,
                schema=schema,
                lexical_graph_config=self._lexical_cfg,
            )
            _normalize_node_names(graph)
            await self._writer.run(graph=graph, lexical_graph_config=self._lexical_cfg)
            print(f"  Graph: {len(graph.nodes)} nodes, {len(graph.relationships)} rels")

    async def resolve(self) -> None:
        """Merge duplicate :__Entity__ nodes (exact-match then embedding-based)."""

        exact_resolver = SinglePropertyExactMatchResolver(
            driver=self._driver, resolve_property="name"
        )
        stats = await exact_resolver.run()
        print(f"\nExact-match resolution: {stats}")

        embedding_resolver = AzureEmbeddingResolver(
            driver=self._driver,
            embedder=self._raw_embedder,
            resolve_properties=["name", "description"],
        )
        emb_stats = await embedding_resolver.run()
        print(f"Embedding resolution:   {emb_stats}")



        with self._driver.session() as session:
            residual = session.run(
                """
                MATCH (n:__Entity__)
                WHERE n.name IS NOT NULL
                WITH labels(n) AS lbls, n.name AS name, count(*) AS c
                WHERE c > 1
                RETURN lbls, name, c ORDER BY c DESC LIMIT 10
                """
            ).data()
        if residual:
            print(f"WARNING: residual duplicates after resolver: {residual}")
        else:
            print("No residual duplicates.")

    def _ensure_indexes(self) -> None:
        with self._driver.session() as session:
            session.run(
                """
                CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
                FOR (c:Chunk) ON c.embedding
                OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}}
                """
            )
            session.run(
                """
                CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
                FOR (c:Chunk) ON EACH [c.text]
                """
            )
        print("Indexes ensured.")

    async def clean(self) -> None:
        """Utility to clear the graph before ingestion."""

        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

        print("Graph cleared.")

    async def run(self) -> None:

        self._ensure_indexes()
        await self.clean()

        for loader in self._loaders:
            await self.ingest(loader)

        await self.resolve()

