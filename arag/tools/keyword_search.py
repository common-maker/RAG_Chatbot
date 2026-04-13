"""arag/tools/keyword_search.py — Exact keyword matching over chunk JSON."""
from pydantic import BaseModel, Field
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken
    _TOKENIZER = tiktoken.encoding_for_model("gpt-4o")
except ImportError:
    raise ImportError("tiktoken is required. Install with: pip install tiktoken")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _load_chunks(chunks_file: str) -> List[Dict[str, Any]]:
    """
    Load chunks from a JSON file.

    Accepted formats:
      - List of dicts with at least 'id' and 'text' keys  (preferred).
      - List of "id:text" strings                          (legacy fallback).
    """
    path = Path(chunks_file)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return []

    if isinstance(data[0], dict):
        # Validate required keys
        for i, item in enumerate(data):
            if "id" not in item or "text" not in item:
                raise ValueError(
                    f"Chunk at index {i} is missing 'id' or 'text' key."
                )
        return data

    # Legacy: "id:text" strings
    chunks = []
    for item in data:
        if isinstance(item, str) and ":" in item:
            chunk_id, text = item.split(":", 1)
            chunks.append({"id": chunk_id.strip(), "text": text})
    return chunks

class KeywordSearchArgs(BaseModel):
    keywords: List[str] = Field(..., description="List of 1–3-word keywords (e.g. ['Einstein', 'relativity', '1905']).")
    top_k: int = Field(default=5, description="Top-ranked chunks to return (default 5, max 20).")

class KeywordSearchTool(BaseTool):
    """
    Exact keyword search over document chunks loaded from a JSON file.

    Scoring (A-RAG Eq. 1 & 2):
      score(chunk) = Σ count(keyword, chunk) × len(keyword)

    Longer, more specific keywords carry proportionally higher weight.
    """

    def __init__(self, chunks_file: str) -> None:
        self.chunks_file = chunks_file
        self.chunks: List[Dict[str, Any]] = _load_chunks(chunks_file)

    @property
    def name(self) -> str:
        return "keyword_search"
    @property
    def fn_schema(self) -> type[BaseModel]:
        return KeywordSearchArgs

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "keyword_search",
                "description": (
                    "Search for document chunks using exact keyword matching "
                    "(case-insensitive). Returns chunk IDs and abbreviated "
                    "sentence snippets where the keywords appear.\n\n"
                    "IMPORTANT: Match keywords literally. Use SHORT, SPECIFIC "
                    "terms (1–3 words each). Each keyword is matched independently.\n\n"
                    "Good keywords: entity names ('Albert Einstein'), technical "
                    "terms ('photosynthesis'), key concepts ('GDP growth').\n\n"
                    "Bad keywords: long phrases, questions, full sentences — "
                    "extract the core noun/term instead.\n\n"
                    "RETURNS: Abbreviated snippets showing where keywords appear. "
                    "Call read_chunk to get full text before answering."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of 1–3-word keywords "
                                "(e.g. ['Einstein', 'relativity', '1905'])."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Top-ranked chunks to return (default 5, max 20).",
                            "default": 5,
                        },
                    },
                    "required": ["keywords"],
                },
            },
        }

    def _extract_snippet_sentences(
        self, text: str, keywords: List[str], max_sentences: int = 5
    ) -> List[str]:
        """
        Extract sentences that contain at least one keyword.
        Uses a more robust split that handles abbreviations better than a
        simple punctuation regex.
        """
        # Split on newlines first (common in parsed documents), then on
        # sentence-ending punctuation followed by whitespace + uppercase.
        raw_sentences = re.split(r'\n+|(?<=[.!?])\s+(?=[A-Z])', text.strip())
        matched = []
        for s in raw_sentences:
            s = s.strip()
            if not s:
                continue
            if any(kw.lower() in s.lower() for kw in keywords):
                matched.append(s)
            if len(matched) >= max_sentences:
                break
        return matched

    def execute(
        self,
        context: "AgentContext",
        keywords: List[str],
        top_k: int = 5,
    ) -> Tuple[str, Dict[str, Any]]:
        top_k = min(top_k, 20)

        if not keywords:
            return "Error: no keywords provided.", {"retrieved_tokens": 0, "chunks_found": 0}

        scored: List[Dict[str, Any]] = []
        for chunk in self.chunks:
            text = chunk["text"]
            text_lower = text.lower()
            chunk_id = str(chunk["id"])

            matched_keywords = []
            total_score = 0
            for kw in keywords:
                kw_lower = kw.lower()
                count = text_lower.count(kw_lower)
                if count > 0:
                    matched_keywords.append(kw)
                    total_score += count * len(kw)

            if total_score > 0:
                snippet_sentences = self._extract_snippet_sentences(text, matched_keywords)
                scored.append(
                    {
                        "chunk_id": chunk_id,
                        "score": total_score,
                        "snippet_sentences": snippet_sentences,
                        "keywords_found": matched_keywords,
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        top_chunks = scored[:top_k]

        if not top_chunks:
            result = f"No chunks matched the keywords: {keywords}"
            context.add_retrieval_log(
                "keyword_search", tokens=0,
                metadata={"keywords": keywords, "chunks_found": 0},
            )
            return result, {"retrieved_tokens": 0, "chunks_found": 0}

        result_parts: List[str] = []
        all_snippet_text = []
        for item in top_chunks:
            if item["snippet_sentences"]:
                snippet = "... " + " ... ".join(item["snippet_sentences"]) + " ..."
                all_snippet_text.extend(item["snippet_sentences"])
            else:
                snippet = "(keywords found but no extractable sentence match)"
            result_parts.append(
                f"Chunk ID: {item['chunk_id']} | score={item['score']} | "
                f"keywords={item['keywords_found']}\n{snippet}"
            )

        tool_result = "\n\n".join(result_parts)
        retrieved_tokens = _count_tokens("\n".join(all_snippet_text)) if all_snippet_text else 0

        context.add_retrieval_log(
            "keyword_search",
            tokens=retrieved_tokens,
            metadata={
                "keywords": keywords,
                "chunks_found": len(top_chunks),
                "chunk_ids": [c["chunk_id"] for c in top_chunks],
            },
        )

        return tool_result, {"retrieved_tokens": retrieved_tokens, "chunks_found": len(top_chunks)}
