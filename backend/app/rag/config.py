from dataclasses import dataclass
from pathlib import Path
from app.config import (
    KNOWLEDGE_BASE_DIR,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_TOP_K,
    RAG_VECTOR_DB_PATH,
)


@dataclass
class RAGConfig:
    knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR
    vector_db_path: Path = RAG_VECTOR_DB_PATH
    chunk_size: int = RAG_CHUNK_SIZE
    chunk_overlap: int = RAG_CHUNK_OVERLAP
    top_k: int = RAG_TOP_K
    min_similarity_threshold: float = 0.05
