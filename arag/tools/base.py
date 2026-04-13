"""arag/tools/base.py — Abstract base class for all A-RAG tools."""
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from arag.core.context import AgentContext


class BaseTool(ABC):
    """
    Abstract base class for all A-RAG retrieval tools.

    Every concrete tool must implement:
      - name       : unique string identifier used by the ToolRegistry.
      - get_schema : OpenAI function-calling JSON schema.
      - execute    : run the tool, return (result_str, log_dict).
    """
    @property
    @abstractmethod
    def fn_schema(self) -> type[BaseModel]:
        """Return the Pydantic model representing the tool's arguments."""
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name (matches the function name in get_schema)."""

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return an OpenAI-compatible function-calling schema dict."""

    @abstractmethod
    def execute(
        self, context: "AgentContext", **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute the tool.

        Parameters
        ----------
        context : AgentContext
            Shared per-query state (chunk tracker, retrieval log).
        **kwargs
            Tool-specific arguments as defined in get_schema.

        Returns
        -------
        result : str
            Human-readable result string passed back to the LLM.
        log : dict
            Structured metadata for auditing (token counts, chunk IDs, etc.).
        """
