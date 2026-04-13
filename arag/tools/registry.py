"""arag/tools/registry.py — Central registry for all A-RAG tools."""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext


class ToolRegistry:
    """
    Registry that holds all BaseTool instances and routes execution calls.

    Usage
    -----
    registry = ToolRegistry()
    registry.register(KeywordSearchTool(...))
    registry.register(SemanticSearchTool(...))
    registry.register(ReadChunkTool(...))

    # Get OpenAI-compatible schemas for all tools
    schemas = registry.get_all_schemas()

    # Execute a tool by name
    result, log = registry.execute("keyword_search", context, keywords=["AI"])
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """Register a tool. Raises ValueError if name is already taken."""
        if tool.name in self._tools:
            raise ValueError(
                f"A tool named '{tool.name}' is already registered. "
                "Unregister it first or use a different name."
            )
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name. Silently ignored if name not found."""
        self._tools.pop(name, None)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> BaseTool | None:
        """Return the tool with the given name, or None."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI function-calling schemas for every registered tool."""
        return [tool.get_schema() for tool in self._tools.values()]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self, name: str, context: "AgentContext", **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute a tool by name.

        Returns
        -------
        result : str
            The tool's human-readable output.
        log : dict
            Structured metadata (token counts, chunk IDs, etc.).
        """
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self.list_tools()) or "none"
            return (
                f"Error: tool '{name}' not found. Available tools: {available}.",
                {"error": "tool_not_found", "requested": name},
            )

        try:
            return tool.execute(context, **kwargs)
        except TypeError as e:
            # Bad kwargs — surface a helpful error instead of a raw traceback.
            return (
                f"Error: incorrect arguments for tool '{name}': {e}",
                {"error": "bad_arguments", "detail": str(e)},
            )
        except Exception as e:
            return (
                f"Error executing tool '{name}': {e}",
                {"error": "execution_error", "detail": str(e)},
            )
