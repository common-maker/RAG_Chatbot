"""
Chainlit application for A-RAG Chatbot (arXiv:2602.03442).

Changes from previous version:
- arag_tools replaced by context_holder (ContextHolder from engine.py).
- context_holder.reset() clears the AgentContext at the start of each query.
- Source attribution reads context_holder.read_chunks (same API, cleaner name).
- Streaming fallback for agents that don't support async_response_gen.
"""

import chainlit as cl
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.openai import OpenAIEmbedding

from engine import get_chat_engine
from logger import setup_logger

logger = setup_logger(__name__, "app.log")

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")


@cl.on_chat_start
async def start():
    logger.info("New chat session started.")
    Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])
    await cl.Message(content="🔄 Initialising A-RAG engine…").send()

    try:
        agent, context_holder = get_chat_engine()
        cl.user_session.set("agent", agent)
        cl.user_session.set("context_holder", context_holder)

        await cl.Message(
            content=(
                "✅ **A-RAG Ready!**\n\n"
                "I answer questions about your documents using multi-step retrieval — "
                "searching, reading, and reasoning over the full corpus before responding.\n\n"
                "Ask me anything about your documents."
            )
        ).send()

    except FileNotFoundError as e:
        logger.error(f"Engine init failed (missing files): {e}")
        await cl.Message(
            content=(
                f"❌ **Initialisation failed:** {e}\n\n"
                "Run `DynamicSectionRetrieverIngestion.py` to ingest your documents first."
            )
        ).send()

    except Exception as e:
        logger.error(f"Engine init failed: {e}", exc_info=True)
        await cl.Message(content=f"❌ Initialisation error: {e}").send()


@cl.on_message
async def main(message: cl.Message):
    logger.info(f"Received: {message.content}")

    agent = cl.user_session.get("agent")
    context_holder = cl.user_session.get("context_holder")

    if not agent:
        await cl.Message(
            content="❌ Agent not initialised. Please refresh the page."
        ).send()
        return

    # Reset per-query state: clears chunk-read tracker and retrieval log.
    if context_holder:
        context_holder.reset()

    msg = cl.Message(content="", author="A-RAG Assistant")

    try:
        response = await agent.astream_chat(message.content)

        # Stream answer tokens; fall back to a direct send if streaming is unavailable.
        if hasattr(response, "async_response_gen"):
            async for token in response.async_response_gen():
                await msg.stream_token(token)
        else:
            response_text = getattr(response, "response", str(response))
            await msg.stream_token(response_text)

        # Append source attribution: which chunks were fully read.
        if context_holder and context_holder.read_chunks:
            read_ids = sorted(
                context_holder.read_chunks,
                key=lambda x: int(x) if x.isdigit() else x,
            )
            max_shown = 10
            shown = read_ids[:max_shown]
            overflow = len(read_ids) - max_shown

            sources_text = "\n\n---\n**📚 Chunks Read:**\n" + "\n".join(
                f"- Chunk {cid}" for cid in shown
            )
            if overflow > 0:
                sources_text += f"\n- … and {overflow} more"

            await msg.stream_token(sources_text)

        await msg.send()

    except Exception as e:
        logger.error(f"Error during chat: {e}", exc_info=True)
        if not msg.content:
            await cl.Message(content=f"❌ Error: {e}").send()
        else:
            await msg.stream_token(f"\n\n❌ Error: {e}")
            await msg.send()
