import json
import math
from pathlib import Path
from typing import Any

from app.rag.loader import DocumentChunk


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot_product / (mag1 * mag2)


class VectorStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path
        self.chunks: list[dict[str, Any]] = []

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        for chunk, embedding in zip(chunks, embeddings):
            self.chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "embedding": embedding,
                }
            )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        min_score: float = 0.05,
    ) -> list[dict[str, Any]]:
        results = []
        for item in self.chunks:
            score = cosine_similarity(query_embedding, item["embedding"])
            if score >= min_score:
                results.append(
                    {
                        "chunk_id": item["chunk_id"],
                        "text": item["text"],
                        "metadata": item["metadata"],
                        "score": score,
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self, target_path: Path | None = None) -> None:
        path = target_path or self.db_path
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.chunks, indent=2), encoding="utf-8")

    def load(self, target_path: Path | None = None) -> bool:
        path = target_path or self.db_path
        if not path or not path.exists():
            return False
        try:
            content = path.read_text(encoding="utf-8")
            self.chunks = json.loads(content)
            return True
        except (json.JSONDecodeError, OSError):
            self.chunks = []
            return False
