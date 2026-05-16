from langchain_openai import AzureOpenAIEmbeddings
import neo4j
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.types import RetrieverResultItem
import json
from sentence_transformers import CrossEncoder

class CustomKnowledgeGraphRetriever(HybridCypherRetriever):
    def __init__(self, 
                 driver: neo4j.Driver, 
                 embedder: AzureOpenAIEmbeddings,
                 reranker: CrossEncoder,
                 retrieval_query: str = None,
                 result_formatter: callable = None,
                 vector_index_name: str = "chunk_embedding",
                 fulltext_index_name: str = "chunk_fulltext",
                 ) -> None:
        
        self._driver = driver
        self._embedder = embedder
        self._reranker = reranker



        super().__init__(
            driver=driver,
            vector_index_name=vector_index_name,
            fulltext_index_name=fulltext_index_name,
            embedder=embedder,
            retrieval_query=retrieval_query or self.retrieval_query,
            result_formatter=result_formatter or CustomKnowledgeGraphRetriever.format_record,
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

    @staticmethod
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

    def build_context(self, retriever_result, max_chars: int = 30000):
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

    def rerank_items(self, query, items, docs=None, top_n=5):
        if not items:
            return []
        pairs = [(query, doc) for doc in (docs or [item.content for item in items])]
        scores = self._reranker.predict(pairs)
        scored = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)
        return [
            (lambda it, s: (it.metadata.__setitem__("rerank_score", float(s)), it)[1])(it, s)
            for it, s in scored[:top_n]
        ]

    def search(self, query_text: str, top_k: int = 20, rerank_top_k: int = 5, execute_reranking: bool = True) -> str:
        results = super().search(query_text=query_text, top_k=top_k)

        if execute_reranking:
            docs_for_rerank = [
                item.metadata.get("seed_text", item.content)[:2000] 
                for item in results.items
            ]

            top_items = self.rerank_items(query_text, results.items, docs=docs_for_rerank, top_n=rerank_top_k)

        context = self.build_context(top_items)

        return context