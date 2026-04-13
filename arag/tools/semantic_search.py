"""arag/tools/semantic_search.py — Sentence-level dense retrieval via local embeddings."""
from pydantic import BaseModel, Field
import os
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np
from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken
    _TOKENIZER = tiktoken.encoding_for_model("gpt-4o")
except ImportError:
    raise ImportError("tiktoken is required. Install with: pip install tiktoken")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is required. Install with: pip install sentence-transformers"
    )


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))

class SemanticSearchArgs(BaseModel):
    query: str = Field(..., description="Natural language query describing the information needed.")
    top_k: int = Field(default=5, description="Number of top results to return (default 5, max 20).")
class SemanticSearchTool(BaseTool):
    """
    Semantic search using sentence-level embeddings (sentence-transformers).

    The index is a pickle file produced by the ingestion pipeline containing:
      - sentences        : List[str]  — individual sentences
      - embeddings       : np.ndarray — L2-normalised sentence embeddings (N × D)
      - sentence_to_chunk: List[str]  — parallel mapping sentence → chunk_id
      - chunks           : Dict[str, Dict] — {chunk_id: {"text": str, ...}}

    Retrieval strategy (A-RAG Eq. 3):
      1. Encode the query with the same model used at index time.
      2. Compute cosine similarity via dot product (embeddings are normalised).
      3. For each chunk that appears in top-K × 3 results, keep the
         highest-scoring sentence.
      4. Return the top-K chunks ranked by that maximum sentence score.
    """

    # Class-level lock: SentenceTransformer.encode is not thread-safe by default.
    _embedding_lock = threading.Lock()

    def __init__(
        self,
        index_dir: str = "index",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.index_dir = index_dir
        self.model_name = model_name
        self.device = device

        self.embedding_model = SentenceTransformer(model_name, device=device)
        self._load_index()

    @property
    def name(self) -> str:
        return "semantic_search"
    @property
    def fn_schema(self) -> type[BaseModel]:
        return SemanticSearchArgs
    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        index_file = Path(self.index_dir) / "sentence_index.pkl"
        if not index_file.exists():
            raise FileNotFoundError(
                f"Sentence index not found at '{index_file}'. "
                "Run DynamicSectionRetrieverIngestion.py to build it."
            )

        with index_file.open("rb") as f:
            data = pickle.load(f)

        required_keys = {"sentences", "embeddings", "sentence_to_chunk", "chunks"}
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(f"Index file is missing keys: {missing}")

        self.sentences: List[str] = data["sentences"]
        self.embeddings: np.ndarray = data["embeddings"]
        self.sentence_to_chunk: List[str] = data["sentence_to_chunk"]
        self.chunks: Dict[str, Dict[str, Any]] = data["chunks"]

        # Validate embedding shape
        if self.embeddings.ndim != 2 or len(self.sentences) != self.embeddings.shape[0]:
            raise ValueError(
                f"Index shape mismatch: {len(self.sentences)} sentences vs "
                f"{self.embeddings.shape[0]} embedding rows."
            )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": (
                    "Find semantically similar passages using sentence-level "
                    "embedding similarity.\n\n"
                    "When to use:\n"
                    "- Exact keywords are unknown or too generic.\n"
                    "- Conceptual / meaning-based matching is needed.\n"
                    "- keyword_search returned no useful results.\n\n"
                    "RETURNS: Abbreviated matched snippets and chunk IDs. "
                    "Call read_chunk for full content before answering."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing the information needed.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top results to return (default 5, max 20).",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self, context: "AgentContext", query: str, top_k: int = 5
    ) -> Tuple[str, Dict[str, Any]]:
        top_k = min(top_k, 20)

        if not query.strip():
            return "Error: empty query.", {"retrieved_tokens": 0, "chunks_found": 0}

        # Thread-safe embedding
        with self._embedding_lock:
            query_vec = self.embedding_model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )[0]

        # Cosine similarity (dot product on normalised vectors)
        similarities: np.ndarray = self.embeddings @ query_vec
        candidate_count = min(top_k * 3, len(self.sentences))
        top_indices = np.argsort(similarities)[::-1][:candidate_count]

        # Aggregate: keep best sentence per chunk
        chunk_best: Dict[str, Dict[str, Any]] = {}
        for idx in top_indices:
            chunk_id = self.sentence_to_chunk[idx]
            sim = float(similarities[idx])
            if chunk_id not in chunk_best or sim > chunk_best[chunk_id]["similarity"]:
                chunk_best[chunk_id] = {
                    "similarity": sim,
                    "sentence": self.sentences[idx],
                }

        ranked = sorted(chunk_best.items(), key=lambda x: x[1]["similarity"], reverse=True)[
            :top_k
        ]

        if not ranked:
            result = f"No semantically relevant passages found for: {query}"
            context.add_retrieval_log(
                "semantic_search", tokens=0, metadata={"query": query, "chunks_found": 0}
            )
            return result, {"retrieved_tokens": 0, "chunks_found": 0}

        result_parts: List[str] = []
        all_snippets: List[str] = []
        for chunk_id, best in ranked:
            snippet = best["sentence"]
            all_snippets.append(snippet)
            result_parts.append(
                f"Chunk ID: {chunk_id} (similarity={best['similarity']:.3f})\n"
                f"Matched: ... {snippet} ..."
            )

        tool_result = "\n\n".join(result_parts)
        retrieved_tokens = _count_tokens("\n".join(all_snippets))

        context.add_retrieval_log(
            "semantic_search",
            tokens=retrieved_tokens,
            metadata={"query": query, "chunks_found": len(ranked)},
        )

        return tool_result, {"retrieved_tokens": retrieved_tokens, "chunks_found": len(ranked)}
