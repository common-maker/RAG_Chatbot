"""arag/tools/read_chunk.py — Full-content chunk retrieval with dedup guard."""
from pydantic import BaseModel, Field
import json
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


def _load_chunks_dict(chunks_file: str) -> Dict[str, str]:
    """Load chunks and return a {chunk_id: text} dict for O(1) lookup."""
    path = Path(chunks_file)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return {}

    if isinstance(data[0], dict):
        return {str(item["id"]): item["text"] for item in data}

    # Legacy: "id:text" strings
    result = {}
    for item in data:
        if isinstance(item, str) and ":" in item:
            chunk_id, text = item.split(":", 1)
            result[chunk_id.strip()] = text
    return result

class ReadChunkArgs(BaseModel):
    chunk_ids: List[str] = Field(..., description="Chunk IDs to retrieve (e.g. ['0', '24', '172']).")
class ReadChunkTool(BaseTool):
    """
    Retrieve the full text of document chunks by their IDs.

    Context tracking (A-RAG Section 3.3):
    Chunks already read during the current query are flagged rather than
    re-sent, preventing duplicate content from bloating the context window.
    """

    def __init__(self, chunks_file: str) -> None:
        self.chunks_file = chunks_file
        self.chunks_dict: Dict[str, str] = _load_chunks_dict(chunks_file)

    @property
    def name(self) -> str:
        return "read_chunk"
    @property
    def fn_schema(self) -> type[BaseModel]:
        return ReadChunkArgs

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "read_chunk",
                "description": (
                    "Read the complete text of document chunks by their IDs.\n\n"
                    "IMPORTANT: Search results (keyword_search / semantic_search) "
                    "only show abbreviated snippets — they are NOT sufficient for "
                    "answering questions. Always call read_chunk on the most "
                    "promising chunk IDs before formulating your answer.\n\n"
                    "Strategy:\n"
                    "- Read all chunks identified as relevant by your searches.\n"
                    "- If the chunk appears truncated or lacks detail, read "
                    "adjacent chunks (chunk_id ± 1).\n"
                    "- Already-read chunks are flagged to avoid redundancy."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Chunk IDs to retrieve (e.g. ['0', '24', '172']).",
                        }
                    },
                    "required": ["chunk_ids"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        chunk_ids: List[str] | None = None,
        chunk_id: str | None = None,   # backward-compat alias
    ) -> Tuple[str, Dict[str, Any]]:
        # Normalise to a list, supporting both the new array arg and the
        # legacy single-string alias.
        if chunk_ids is None:
            if chunk_id is not None:
                chunk_ids = [str(chunk_id)]
            else:
                return "Error: no chunk IDs provided.", {"retrieved_tokens": 0}

        # Deduplicate while preserving request order.
        seen: set = set()
        clean_ids: List[str] = []
        for cid in chunk_ids:
            cid = str(cid).strip()
            if cid and cid not in seen:
                seen.add(cid)
                clean_ids.append(cid)

        if not clean_ids:
            return "Error: all provided chunk IDs were empty.", {"retrieved_tokens": 0}

        result_parts: List[str] = []
        new_chunks_read: List[str] = []
        already_read: List[str] = []
        total_tokens = 0
        sep = "=" * 80

        for cid in clean_ids:
            if context.is_chunk_read(cid):
                already_read.append(cid)
                result_parts.append(
                    f"\n{sep}\n[Chunk {cid}]\n"
                    "This chunk was already read — use the previously retrieved content.\n"
                    f"{sep}"
                )
                continue

            text = self.chunks_dict.get(cid)
            if text is None:
                result_parts.append(f"\n[Chunk {cid}] — Not found in the document store.")
                continue

            chunk_tokens = _count_tokens(text)
            total_tokens += chunk_tokens
            context.mark_chunk_as_read(cid)
            new_chunks_read.append(cid)

            result_parts.append(
                f"\n{sep}\n[Chunk {cid}]\n{'-' * 80}\n{text}\n{sep}"
            )

        tool_result = "\n".join(result_parts)

        context.add_retrieval_log(
            "read_chunk",
            tokens=total_tokens,
            metadata={
                "requested": clean_ids,
                "new_chunks_read": new_chunks_read,
                "already_read": already_read,
            },
        )

        return tool_result, {
            "retrieved_tokens": total_tokens,
            "new_chunks_count": len(new_chunks_read),
            "already_read_count": len(already_read),
        }
