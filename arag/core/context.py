"""
arag/core/context.py — Per-query agent execution context.

AgentContext is created fresh for every user query and carries:
  - The set of chunk IDs that have already been fully read (dedup guard).
  - A structured retrieval log for post-query auditing / UI display.

It is passed into every BaseTool.execute() call so tools can share state
without coupling to each other directly.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class RetrievalEntry:
    """One tool-call entry in the retrieval log."""
    tool_name: str
    tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentContext:
    """
    Mutable execution context for a single user query.

    Lifecycle
    ---------
    Create one instance at the start of each query, pass it to every tool
    call, then inspect it afterwards for logging / UI attribution.

    Attributes
    ----------
    _read_chunks : Set[str]
        Chunk IDs that have been fully read during this query.
    _retrieval_log : List[RetrievalEntry]
        Ordered record of every tool call made during this query.
    """

    def __init__(self) -> None:
        self._read_chunks: Set[str] = set()
        self._retrieval_log: List[RetrievalEntry] = []

    # ------------------------------------------------------------------
    # Chunk-read tracking
    # ------------------------------------------------------------------

    def is_chunk_read(self, chunk_id: str) -> bool:
        """Return True if this chunk has already been fully read."""
        return str(chunk_id) in self._read_chunks

    def mark_chunk_as_read(self, chunk_id: str) -> None:
        """Record that a chunk has been fully read."""
        self._read_chunks.add(str(chunk_id))

    @property
    def read_chunks(self) -> Set[str]:
        """Read-only view of all chunk IDs read so far."""
        return frozenset(self._read_chunks)

    # ------------------------------------------------------------------
    # Retrieval logging
    # ------------------------------------------------------------------

    def add_retrieval_log(
        self,
        tool_name: str,
        tokens: int,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Append a tool-call entry to the retrieval log."""
        self._retrieval_log.append(
            RetrievalEntry(
                tool_name=tool_name,
                tokens=tokens,
                metadata=metadata or {},
            )
        )

    @property
    def retrieval_log(self) -> List[RetrievalEntry]:
        """Ordered list of all retrieval entries for this query."""
        return list(self._retrieval_log)

    @property
    def total_retrieved_tokens(self) -> int:
        """Sum of tokens retrieved across all tool calls."""
        return sum(e.tokens for e in self._retrieval_log)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a compact summary suitable for logging or UI display."""
        return {
            "chunks_read": sorted(self._read_chunks, key=lambda x: int(x) if x.isdigit() else x),
            "total_retrieved_tokens": self.total_retrieved_tokens,
            "tool_calls": [
                {"tool": e.tool_name, "tokens": e.tokens, **e.metadata}
                for e in self._retrieval_log
            ],
        }
