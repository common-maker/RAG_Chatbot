"""
engine.py — A-RAG single-agent chat engine (arXiv:2602.03442).

Architecture
------------
Storage  : flat JSON (chunks) + pickle (sentence embeddings via sentence-transformers)
Tools    : KeywordSearchTool / SemanticSearchTool / ReadChunkTool — all extend BaseTool
           and are managed by ToolRegistry.
Agent    : LlamaIndex ReActAgent. Each BaseTool is wrapped in a thin async FunctionTool
           adapter so the LlamaIndex agent can call it transparently.
Context  : AgentContext is created per query and passed through every tool call
           to track chunk-read state and retrieval logs.
"""

from pathlib import Path
from typing import Optional, Tuple

from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from arag.core.context import AgentContext
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.semantic_search import SemanticSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.tools.registry import ToolRegistry
from arag.tools.base import BaseTool

from config import Config
from logger import setup_logger

logger = setup_logger(__name__, "engine.log")

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")


# ============================================================
# Per-query context holder
# ============================================================

class ContextHolder:
    """
    Thin mutable wrapper around AgentContext.

    The FunctionTool closures capture this holder by reference. Calling
    reset() between queries swaps in a fresh AgentContext without needing
    to recreate the tools or the agent.
    """

    def __init__(self) -> None:
        self.ctx: AgentContext = AgentContext()

    def reset(self) -> None:
        """Replace the inner context — call at the start of every new query."""
        self.ctx = AgentContext()

    @property
    def read_chunks(self):
        return self.ctx.read_chunks

    @property
    def retrieval_summary(self):
        return self.ctx.summary()


# ============================================================
# BaseTool → LlamaIndex FunctionTool adapter
# ============================================================

# def _wrap_tool(tool: BaseTool, holder: ContextHolder) -> FunctionTool:
#     """
#     Wrap a BaseTool as an async LlamaIndex FunctionTool.

#     The async wrapper:
#       1. Reads the current AgentContext from the holder.
#       2. Calls tool.execute(context, **kwargs) synchronously (tools are CPU-bound).
#       3. Returns the result string to the ReActAgent.
#       4. The log dict is captured inside AgentContext for post-query inspection.
#     """
#     schema_fn = tool.get_schema()["function"]
#     description = tool.get_schema()["function"]["description"]

#     async def _async_fn(**kwargs) -> str:
#         result, _ = tool.execute(holder.ctx, **kwargs)
#         return result

#     _async_fn.__name__ = tool.name

#     return FunctionTool.from_defaults(
#         async_fn=_async_fn,
#         name=tool.name,
#         description=description,
#     )

def _wrap_tool(tool: BaseTool, holder: ContextHolder) -> FunctionTool:
    """
    Wrap a BaseTool as an async LlamaIndex FunctionTool.

    The async wrapper:
      1. Reads the current AgentContext from the holder.
      2. Calls tool.execute(context, **kwargs) synchronously (tools are CPU-bound).
      3. Returns the result string to the ReActAgent.
      4. The log dict is captured inside AgentContext for post-query inspection.
    """
    # Extract the description from your existing schema dictionary
    description = tool.get_schema()["function"]["description"]

    async def _async_fn(**kwargs) -> str:
        result, _ = tool.execute(holder.ctx, **kwargs)
        return result

    # LlamaIndex uses the function name to register the tool
    _async_fn.__name__ = tool.name

    return FunctionTool.from_defaults(
        async_fn=_async_fn,
        name=tool.name,
        description=description,
        # THIS IS THE CRITICAL FIX: 
        # Pass the Pydantic schema so LlamaIndex knows exactly what arguments to expect
        fn_schema=tool.fn_schema 
    )



# ============================================================
# Chat Engine Factory
# ============================================================

class ChatEngineFactory:
    def __init__(
        self,
        chunks_file: Optional[str] = None,
        index_dir: Optional[str] = None,
    ):
        self.chunks_file = chunks_file or Config.CHUNKS_FILE
        self.index_dir = index_dir or Config.INDEX_DIR

    def _build_registry(self) -> ToolRegistry:
        """Instantiate all three A-RAG tools and register them."""
        if not Path(self.chunks_file).exists():
            raise FileNotFoundError(
                f"Chunks file not found at '{self.chunks_file}'. "
                "Run DynamicSectionRetrieverIngestion.py first."
            )

        registry = ToolRegistry()

        registry.register(KeywordSearchTool(chunks_file=self.chunks_file))
        logger.info("Registered: keyword_search")

        registry.register(
            SemanticSearchTool(
                index_dir=self.index_dir,
                model_name=getattr(Config, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                device=getattr(Config, "EMBEDDING_DEVICE", None),
            )
        )
        logger.info("Registered: semantic_search")

        registry.register(ReadChunkTool(chunks_file=self.chunks_file))
        logger.info("Registered: read_chunk")

        return registry

    def create_chat_engine(
        self,
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> Tuple[ReActAgent, ContextHolder]:
        # Explicit None-checks so temperature=0.0 is respected.
        if llm_model is None:
            llm_model = Config.LLM_MODEL
        if llm_temperature is None:
            llm_temperature = Config.LLM_TEMPERATURE
        if system_prompt is None:
            system_prompt = Config.SYSTEM_PROMPT

        registry = self._build_registry()

        # One holder shared across all tool closures; reset per query in the app.
        holder = ContextHolder()

        function_tools = [
            _wrap_tool(registry.get(name), holder)
            for name in registry.list_tools()
        ]

        llm = OpenAI(model=llm_model, temperature=llm_temperature)
        memory = ChatMemoryBuffer.from_defaults(token_limit=Config.MEMORY_TOKEN_LIMIT)

        agent_prompt = (
            system_prompt + "\n\n"
            "# Strategy\n"
            "Work iteratively: search → read → evaluate → search → read → … → answer. "
            "For multi-hop questions, decompose into sub-questions and address each step by step.\n\n"
            "# Rules\n"
            "- Always call read_chunk on the most relevant chunk IDs before answering.\n"
            "- Ground every claim in retrieved document content.\n"
            "- Cite the specific [Chunk ID] that supports each statement.\n"
            "- Avoid speculation beyond what the documents explicitly state.\n"
            "- If no relevant content is found after a thorough search, say so clearly."
        )

        agent = ReActAgent.from_tools(
            tools=function_tools,
            llm=llm,
            memory=memory,
            system_prompt=agent_prompt,
            verbose=Config.ENABLE_VERBOSE,
            max_iterations=Config.MAX_ITERATIONS,
        )

        logger.info("A-RAG single-agent engine created successfully.")
        return agent, holder


# ============================================================
# Public factory function
# ============================================================

def get_chat_engine(
    chunks_file: Optional[str] = None,
    index_dir: Optional[str] = None,
) -> Tuple[ReActAgent, ContextHolder]:
    """
    Build and return (agent, context_holder).

    Call context_holder.reset() at the start of every user query to clear the
    per-query chunk-read tracker (A-RAG Section 3.3).
    """
    factory = ChatEngineFactory(chunks_file=chunks_file, index_dir=index_dir)
    return factory.create_chat_engine()


if __name__ == "__main__":
    try:
        agent, _ = get_chat_engine()
        logger.info("A-RAG engine test successful.")
    except Exception as e:
        logger.error(f"Engine test failed: {e}")
