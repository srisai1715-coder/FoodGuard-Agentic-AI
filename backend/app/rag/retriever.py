from typing import Any
from app.rag.config import RAGConfig
from app.rag.embeddings import EmbeddingGenerator
from app.rag.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        config: RAGConfig | None = None,
        vector_store: VectorStore | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        self.vector_store = vector_store or VectorStore(db_path=self.config.vector_db_path)
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

        # Load store if vector_store was not populated yet
        if not self.vector_store.chunks:
            self.vector_store.load()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip() or not self.vector_store.chunks:
            return []

        limit = top_k or self.config.top_k
        query_embedding = self.embedding_generator.embed_text(query)
        results = self.vector_store.similarity_search(
            query_embedding,
            top_k=limit,
            min_score=self.config.min_similarity_threshold,
        )
        return results

    def format_rag_context(self, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return ""

        formatted_snippets = []
        for idx, chunk in enumerate(chunks, 1):
            title = chunk.get("metadata", {}).get("title", "Reference Guide")
            source = chunk.get("metadata", {}).get("source_file", "knowledge_base")
            text = chunk.get("text", "").strip()
            formatted_snippets.append(
                f"[Source {idx}: {title} ({source})]\n{text}"
            )

        return "\n\n".join(formatted_snippets)

    def extract_sources(self, chunks: list[dict[str, Any]]) -> list[str]:
        sources = []
        for chunk in chunks:
            title = chunk.get("metadata", {}).get("title")
            file = chunk.get("metadata", {}).get("source_file")
            if title and file:
                sources.append(f"{title} ({file})")
            elif title:
                sources.append(title)
        return list(dict.fromkeys(sources))  # Remove duplicates while preserving order
