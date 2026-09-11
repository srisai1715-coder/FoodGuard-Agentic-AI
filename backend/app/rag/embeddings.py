import math
import re
from collections import Counter
from app.llm.provider import NVIDIAProvider, LLMConfigurationError


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def fallback_embedding(text: str, vector_dim: int = 128) -> list[float]:
    tokens = _tokenize(text)
    if not tokens:
        return [0.0] * vector_dim

    counts = Counter(tokens)
    vec = [0.0] * vector_dim
    for token, count in counts.items():
        # Hash token into vector dimension index
        idx = hash(token) % vector_dim
        vec[idx] += float(count)

    # L2 Normalize
    magnitude = math.sqrt(sum(v * v for v in vec))
    if magnitude > 0:
        vec = [v / magnitude for v in vec]
    return vec


class EmbeddingGenerator:
    def __init__(self, provider: NVIDIAProvider | None = None) -> None:
        self.provider = provider
        self.use_fallback = False

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * 128

        if self.use_fallback or not self.provider:
            return fallback_embedding(text)

        try:
            # Try calling OpenAI embedding if client supports it
            client = getattr(self.provider, "client", None)
            if client and hasattr(client, "embeddings"):
                response = client.embeddings.create(
                    input=text,
                    model="nvidia/nv-embedqa-e5-v5",
                )
                return response.data[0].embedding
        except (AttributeError, LLMConfigurationError, Exception):
            self.use_fallback = True

        return fallback_embedding(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]
