from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from app.graph.workflow import build_workflow
from app.rag.config import RAGConfig
from app.rag.embeddings import EmbeddingGenerator
from app.rag.loader import chunk_text, load_documents
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore, cosine_similarity


def test_document_loading(tmp_path: Path):
    doc_path = tmp_path / "test_doc.md"
    doc_path.write_text(
        "# Test Food Safety Document\n\n"
        "Milk and eggs are major allergens. Keep food stored at low temperatures.",
        encoding="utf-8",
    )

    chunks = load_documents(tmp_path, chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 1
    assert chunks[0].metadata["title"] == "Test Food Safety Document"
    assert chunks[0].metadata["source_file"] == "test_doc.md"
    assert "Milk and eggs" in chunks[0].text


def test_text_chunking():
    text = "Paragraph 1: Food safety standards.\n\nParagraph 2: Allergen controls.\n\nParagraph 3: Temperature rules."
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    assert any("Paragraph 1" in c for c in chunks)
    assert any("Paragraph 2" in c for c in chunks)


def test_embedding_and_index_creation():
    embedder = EmbeddingGenerator()
    vec1 = embedder.embed_text("Milk allergen guide")
    vec2 = embedder.embed_text("Milk allergen guide")
    vec3 = embedder.embed_text("Totally unrelated text about rocket space travel")

    assert len(vec1) == 128
    sim1 = cosine_similarity(vec1, vec2)
    sim2 = cosine_similarity(vec1, vec3)

    assert sim1 > 0.99
    assert sim1 > sim2


def test_retrieval(tmp_path: Path):
    doc_path = tmp_path / "fda_guide.md"
    doc_path.write_text(
        "# FDA Food Allergen Guide\n\n"
        "Under the FASTER Act, sesame is classified as a major food allergen alongside milk, eggs, peanuts, and wheat.",
        encoding="utf-8",
    )

    db_path = tmp_path / "index.json"
    chunks = load_documents(tmp_path)
    embedder = EmbeddingGenerator()
    embeddings = embedder.embed_documents([c.text for c in chunks])

    store = VectorStore(db_path=db_path)
    store.add_chunks(chunks, embeddings)
    store.save()

    cfg = RAGConfig(knowledge_base_dir=tmp_path, vector_db_path=db_path, top_k=2)
    retriever = Retriever(config=cfg, vector_store=store, embedding_generator=embedder)

    results = retriever.retrieve("sesame allergen")
    assert len(results) > 0
    assert "sesame" in results[0]["text"].lower()


def test_source_metadata_preservation(tmp_path: Path):
    doc_path = tmp_path / "dietary_rules.md"
    doc_path.write_text(
        "# USDA Sodium Rules\n\nDaily sodium intake limit is 2300mg per day.",
        encoding="utf-8",
    )

    chunks = load_documents(tmp_path)
    embedder = EmbeddingGenerator()
    embeddings = embedder.embed_documents([c.text for c in chunks])
    store = VectorStore()
    store.add_chunks(chunks, embeddings)

    retriever = Retriever(vector_store=store, embedding_generator=embedder)
    results = retriever.retrieve("sodium limit")

    assert len(results) > 0
    metadata = results[0]["metadata"]
    assert metadata["title"] == "USDA Sodium Rules"
    assert metadata["source_file"] == "dietary_rules.md"

    sources = retriever.extract_sources(results)
    assert "USDA Sodium Rules (dietary_rules.md)" in sources


def test_no_result_retrieval(tmp_path: Path):
    db_path = tmp_path / "empty_index.json"
    store = VectorStore(db_path=db_path)
    cfg = RAGConfig(knowledge_base_dir=tmp_path, vector_db_path=db_path)
    retriever = Retriever(config=cfg, vector_store=store)

    results = retriever.retrieve("")
    assert results == []

    results_unrelated = retriever.retrieve("xyz123randomnonexistentquery")
    # Low score query should return empty or low results
    assert isinstance(results_unrelated, list)


def test_rag_langgraph_integration():
    mock_provider = MagicMock()
    mock_provider.generate_response.side_effect = [
        '{"intent": "GENERAL_QUESTION", "confidence": 0.95, "reason": "General question"}',
        "According to the FDA Food Allergen Guide [Source: FDA Food Allergen Guide], sesame is a major allergen.",
    ]

    app = build_workflow(provider=mock_provider)
    result = app.invoke({"messages": [HumanMessage(content="Is sesame a major allergen?")]})

    assert "messages" in result
    assert len(result["messages"]) >= 1
    assert "rag_context" in result
    assert "rag_sources" in result
    assert isinstance(result["rag_sources"], list)


def test_sesame_allergen_general_question_rag_routing():
    mock_provider = MagicMock()
    mock_provider.generate_response.side_effect = [
        '{"intent": "GENERAL_QUESTION", "confidence": 0.95, "reason": "General educational question"}',
        "Sesame is considered a major food allergen under the FASTER Act of 2021 [Source: FDA Major Food Allergens and Allergen Labeling Guide].",
    ]

    app = build_workflow(provider=mock_provider)
    result = app.invoke({"messages": [HumanMessage(content="Why is sesame considered a major food allergen?")]})

    assert "messages" in result
    last_msg = result["messages"][-1].content
    assert "Sesame is considered a major food allergen" in last_msg
    assert "I can check the product for common allergens, but I need the ingredient list" not in last_msg


def test_sesame_allergen_safety_intent_rag_fallback():
    mock_provider = MagicMock()
    # Test case where Manager classifies as SAFETY_ALLERGEN even without product data
    mock_provider.generate_response.side_effect = [
        '{"intent": "SAFETY_ALLERGEN", "confidence": 0.95, "reason": "Allergen question"}',
        "Sesame is considered a major food allergen under the FASTER Act of 2021 [Source: FDA Major Food Allergens and Allergen Labeling Guide].",
    ]

    app = build_workflow(provider=mock_provider)
    result = app.invoke({"messages": [HumanMessage(content="Why is sesame considered a major food allergen?")]})

    assert "messages" in result
    last_msg = result["messages"][-1].content
    assert "Sesame is considered a major food allergen" in last_msg
    assert "I can check the product for common allergens, but I need the ingredient list" not in last_msg



