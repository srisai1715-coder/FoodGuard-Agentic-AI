import os
from pathlib import Path


def _load_local_env() -> None:
    """
    Load values from the project's local .env file when running
    outside Docker Compose.

    Existing environment variables always take precedence.
    """
    env_file = Path(__file__).resolve().parents[2] / ".env"

    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        name, value = line.split("=", 1)

        os.environ.setdefault(
            name.strip(),
            value.strip().strip('"').strip("'"),
        )


_load_local_env()


# ---------------------------------------------------------------------------
# NVIDIA LLM Configuration
# ---------------------------------------------------------------------------

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "")
NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)


# ---------------------------------------------------------------------------
# Open Food Facts Configuration
# ---------------------------------------------------------------------------

OPEN_FOOD_FACTS_BASE_URL = os.getenv(
    "OPEN_FOOD_FACTS_BASE_URL",
    "https://world.openfoodfacts.org/api/v2",
).rstrip("/")

FOODGUARD_USER_AGENT = os.getenv(
    "FOODGUARD_USER_AGENT",
    "FoodGuard-AI/1.0",
)

OPEN_FOOD_FACTS_TIMEOUT = float(
    os.getenv("OPEN_FOOD_FACTS_TIMEOUT", "10")
)


# ---------------------------------------------------------------------------
# OCR Configuration
# ---------------------------------------------------------------------------

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")


# ---------------------------------------------------------------------------
# Database Configuration
# ---------------------------------------------------------------------------
#
# Docker Compose supplies DATABASE_URL inside the backend container.
#
# For local development, DATABASE_URL can be provided through .env.
# We intentionally do NOT keep a hardcoded database password here.
#

DATABASE_URL = os.getenv("DATABASE_URL", "")


# ---------------------------------------------------------------------------
# RAG Configuration
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_DIR = Path(
    os.getenv(
        "KNOWLEDGE_BASE_DIR",
        str(ROOT_DIR / "data" / "knowledge_base"),
    )
)

RAG_VECTOR_DB_PATH = Path(
    os.getenv(
        "RAG_VECTOR_DB_PATH",
        str(ROOT_DIR / "data" / "vector_db" / "knowledge_index.json"),
    )
)

RAG_CHUNK_SIZE = int(
    os.getenv("RAG_CHUNK_SIZE", "500")
)

RAG_CHUNK_OVERLAP = int(
    os.getenv("RAG_CHUNK_OVERLAP", "100")
)

RAG_TOP_K = int(
    os.getenv("RAG_TOP_K", "3")
)