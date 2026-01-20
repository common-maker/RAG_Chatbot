import chainlit as cl
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from engine import get_chat_engine  # Updated import

# GLOBAL SETTINGS
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

@cl.on_chat_start
async def start():
    Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])
    
    try:
        # Load the CHAT engine (not just query engine)
        chat_engine = get_chat_engine()
        
        # Store it in the session. 
        # Crucial: The 'chat_engine' object holds the conversation memory inside it.
        cl.user_session.set("chat_engine", chat_engine)
        
        await cl.Message(content="Ready! I can now remember our conversation.").send()
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()

@cl.on_message
async def main(message: cl.Message):
    chat_engine = cl.user_session.get("chat_engine")
    msg = cl.Message(content="", author="Assistant")
    
    # 1. Use 'astream_chat' for conversational streaming
    # This automatically adds 'message.content' to the internal memory
    response = await chat_engine.astream_chat(message.content)

    # 2. Stream the response tokens
    async for token in response.async_response_gen():
        await msg.stream_token(token)
        
    # 3. Debug: Show Sources
    # Note: In ChatEngine, sources are still available in source_nodes
    if response.source_nodes:
        # Just grabbing the first one for brevity
        context_len = len(response.source_nodes[0].node.get_content())
        await msg.stream_token(f"\n\n*(Context size: {context_len} chars)*")

    await msg.send()