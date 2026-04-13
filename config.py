"""
config.py — Central configuration for the A-RAG project.

All runtime tunables live here. Override values via environment variables
(prefixed ARAG_) or by editing the defaults directly.

Environment variable mapping
-----------------------------
ARAG_LLM_MODEL              → Config.LLM_MODEL
ARAG_LLM_TEMPERATURE        → Config.LLM_TEMPERATURE
ARAG_MAX_ITERATIONS         → Config.MAX_ITERATIONS
ARAG_MEMORY_TOKEN_LIMIT     → Config.MEMORY_TOKEN_LIMIT
ARAG_SIMILARITY_TOP_K       → Config.SIMILARITY_TOP_K
ARAG_ENABLE_VERBOSE         → Config.ENABLE_VERBOSE         (1/true/yes = True)
ARAG_CHUNKS_FILE            → Config.CHUNKS_FILE
ARAG_INDEX_DIR              → Config.INDEX_DIR
ARAG_EMBEDDING_MODEL        → Config.EMBEDDING_MODEL
ARAG_EMBEDDING_DEVICE       → Config.EMBEDDING_DEVICE       (e.g. "cuda", "cpu")
ARAG_SYSTEM_PROMPT          → Config.SYSTEM_PROMPT
"""

import os
import textwrap
from pathlib import Path


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if raw.isdigit():
        return int(raw)
    return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key, "").strip().lower()
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no"):
        return False
    return default


def _env_optional_str(key: str) -> str | None:
    raw = os.getenv(key, "").strip()
    return raw if raw else None


class Config:
    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    # OpenAI model used by the agent and section-extraction LLM.
    LLM_MODEL: str = _env_str("ARAG_LLM_MODEL", "gpt-4o-mini")

    # Set to 0.0 for fully deterministic answers; raise slightly (e.g. 0.1)
    # for more varied phrasing. Must stay low for a RAG system.
    LLM_TEMPERATURE: float = _env_float("ARAG_LLM_TEMPERATURE", 0.0)

    # Maximum ReAct iterations per query. Raise for complex multi-hop questions.
    MAX_ITERATIONS: int = _env_int("ARAG_MAX_ITERATIONS", 15)

    # Conversation memory budget in tokens. Older turns are evicted when exceeded.
    MEMORY_TOKEN_LIMIT: int = _env_int("ARAG_MEMORY_TOKEN_LIMIT", 8192)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    # Default top-K for keyword_search and semantic_search.
    SIMILARITY_TOP_K: int = _env_int("ARAG_SIMILARITY_TOP_K", 5)

    # ------------------------------------------------------------------
    # Storage paths
    # ------------------------------------------------------------------

    # Flat JSON file produced by ingestion — consumed by KeywordSearchTool
    # and ReadChunkTool at startup.
    CHUNKS_FILE: str = _env_str("ARAG_CHUNKS_FILE", "chunks.json")

    # Directory containing sentence_index.pkl — consumed by SemanticSearchTool.
    INDEX_DIR: str = _env_str("ARAG_INDEX_DIR", "index")

    # Legacy: ChromaDB persist directory (kept for reference; no longer used
    # by the current pipeline which uses JSON + pickle).
    STORAGE_DIR: Path = Path(_env_str("ARAG_STORAGE_DIR", "./chroma_db"))

    # ------------------------------------------------------------------
    # Sentence embedding model (sentence-transformers)
    # ------------------------------------------------------------------

    # Model name accepted by SentenceTransformer(). Swap for a larger or
    # multilingual model if needed (e.g. "BAAI/bge-base-en-v1.5").
    EMBEDDING_MODEL: str = _env_str(
        "ARAG_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    # Inference device passed to SentenceTransformer. None = auto-detect.
    # Use "cuda" to force GPU, "cpu" to force CPU.
    EMBEDDING_DEVICE: str | None = _env_optional_str("ARAG_EMBEDDING_DEVICE")

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------

    # When True the ReActAgent prints its full thought/action/observation
    # trace to stdout — very useful during development.
    ENABLE_VERBOSE: bool = _env_bool("ARAG_ENABLE_VERBOSE", False)


    # Logging Configuration
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
    STORAGE_DIR = BASE_DIR / os.getenv("STORAGE_DIR", "./chroma_db")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_VERBOSE = os.getenv("ENABLE_VERBOSE", "true").lower() == "true"
    LOGS_DIR = BASE_DIR / os.getenv("LOGS_DIR", "logs")

    # ------------------------------------------------------------------
    # Agent system prompt
    # ------------------------------------------------------------------

    SYSTEM_PROMPT: str = textwrap.dedent("""\
        You are a precise, document-grounded question-answering assistant.

        Your knowledge comes exclusively from the document corpus provided
        to you via the retrieval tools. Do not use any external knowledge
        or training data to answer questions.

        When responding:
        - Base every claim on content retrieved from the documents.
        - Cite the specific [Chunk ID] that supports each statement.
        - If the documents do not contain enough information to answer
          confidently, say so explicitly rather than speculating.
        - Keep your answers clear, well-structured, and concise.
    """).strip()

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Create necessary directories
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        
        return True