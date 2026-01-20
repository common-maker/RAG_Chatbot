
import os
import chainlit as cl

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.callbacks import CallbackManager
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# 2. GLOBAL SETTINGS (LlamaIndex v0.10+)
# Configure these once at startup, not inside every user session
Settings.llm = OpenAI(
    model="gpt-4o-mini",  # Updated to a newer/faster model
    temperature=0.1,
    max_tokens=1024
)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.context_window = 4096

@cl.on_chat_start
async def start():
    # 3. Connect Chainlit UI Trace to LlamaIndex
    # This allows you to see the "Steps" in the Chainlit UI
    Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])

    # 4. Load or Build Index
    # Check if storage exists first to avoid try/except block masking other errors
    if os.path.exists("./storage"):
        try:
            storage_context = StorageContext.from_defaults(persist_dir="./storage")
            index = load_index_from_storage(storage_context)
            await cl.Message(content="Loaded index from disk.").send()
        except Exception as e:
            await cl.Message(content=f"Error loading index: {e}").send()
            return
    else:
        # First run: Build index from data
        await cl.Message(content="Building index... this may take a moment.").send()
        if not os.path.exists("./data"):
            os.makedirs("./data") # Safety creation
            
        documents = SimpleDirectoryReader("./data").load_data(show_progress=True)
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir="./storage")
        await cl.Message(content="Index built and saved!").send()

    # 5. Create Query Engine
    # Streaming=True is crucial for a chatty feel
    query_engine = index.as_query_engine(
        streaming=True, 
        similarity_top_k=2
    )
    cl.user_session.set("query_engine", query_engine)

    await cl.Message(
        author="Assistant", 
        content="Hello! I'm ready to chat about your data."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")
    
    msg = cl.Message(content="", author="Assistant")

    # 6. Native Async Query & Streaming
    # Use 'aquery' (async query) instead of wrapping the sync 'query'
    response = await query_engine.aquery(message.content)

    # Stream the tokens as they arrive
    async for token in response.async_response_gen():
        await msg.stream_token(token)

    # 7. (Optional) Show Sources
    # If you want to show WHICH file the answer came from:
    source_names = set()
    for node in response.source_nodes:
        # Get filename from metadata
        file_name = node.metadata.get("file_name", "Unknown Source")
        source_names.add(file_name)
    
    if source_names:
        await msg.stream_token(f"\n\n**Sources:** {', '.join(source_names)}")

    await msg.send()

    