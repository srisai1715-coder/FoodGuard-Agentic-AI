import logging
from pathlib import Path
from app.rag.config import RAGConfig
from app.rag.embeddings import EmbeddingGenerator
from app.rag.loader import load_documents
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest_knowledge_base(config: RAGConfig | None = None) -> int:
    cfg = config or RAGConfig()
    kb_dir = cfg.knowledge_base_dir

    if not kb_dir.exists():
        logger.warning("Knowledge base directory does not exist: %s", kb_dir)
        return 0

    chunks = load_documents(
        directory_path=kb_dir,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    if not chunks:
        logger.warning("No documents found to ingest in %s", kb_dir)
        return 0

    embedder = EmbeddingGenerator()
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_documents(texts)

    store = VectorStore(db_path=cfg.vector_db_path)
    store.add_chunks(chunks, embeddings)
    store.save()

    logger.info(
        "Successfully ingested %d chunks from %s into %s",
        len(chunks),
        kb_dir,
        cfg.vector_db_path,
    )
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingested_count = ingest_knowledge_base()
    print(f"Ingested {ingested_count} knowledge base chunks.")
