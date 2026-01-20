import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

def get_chat_engine(persist_dir="./storage_recursive"):
    """Loads the index and returns a Conversational Chat Engine with Memory."""
    if not os.path.exists(persist_dir):
        raise FileNotFoundError("Storage directory not found. Run ingestion.py first.")

    # 1. Load the Index & Docstore
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    vector_index = load_index_from_storage(storage_context)

    # 2. Setup Your Recursive Retriever (The "Small-to-Big" logic)
    # We maintain the exact same powerful retrieval logic you built before
    vector_retriever = vector_index.as_retriever(similarity_top_k=3)
    retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        node_dict=storage_context.docstore.docs,
        verbose=True
    )

    # 3. Configure Memory
    # This buffer keeps the last ~3000 tokens of conversation history
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    # 4. Create the Chat Engine
    # 'CondensePlusContext' automatically handles the "Rewrite -> Search -> Answer" loop
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=OpenAI(model="gpt-4o-mini", temperature=0.1), # Main answering LLM
        memory=memory,
        system_prompt="You are a helpful assistant who answers based on the retrieved docs.",
        verbose=True # Prints the "Condensing" steps to console
    )

    return chat_engine