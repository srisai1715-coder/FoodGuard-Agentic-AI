import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_title_from_content(content: str, filename: str) -> str:
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return Path(filename).stem.replace("_", " ").title()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[str]:
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    # Split by paragraphs or double newlines first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if not current_chunk:
            current_chunk = paragraph
        elif len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            current_chunk += "\n\n" + paragraph
        else:
            chunks.append(current_chunk)
            # Apply overlap if possible
            overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
            if overlap_text:
                current_chunk = overlap_text + "\n\n" + paragraph
            else:
                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    # Secondary fallback for any oversized chunk
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                end = start + chunk_size
                sub_chunk = chunk[start:end].strip()
                if sub_chunk:
                    final_chunks.append(sub_chunk)
                start += chunk_size - chunk_overlap

    return final_chunks


def load_documents(
    directory_path: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    if not directory_path.exists():
        return chunks

    supported_extensions = {".md", ".txt", ".json"}
    for root, _, files in os.walk(directory_path):
        for file in sorted(files):
            file_path = Path(root) / file
            if file_path.suffix.lower() not in supported_extensions:
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            title = extract_title_from_content(content, file)
            text_chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            for idx, text in enumerate(text_chunks):
                chunk_id = f"{file_path.stem}_chunk_{idx}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=text,
                        metadata={
                            "source_file": file,
                            "source_path": str(file_path),
                            "title": title,
                            "chunk_index": idx,
                            "total_chunks": len(text_chunks),
                        },
                    )
                )

    return chunks
