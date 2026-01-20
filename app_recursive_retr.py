import os
import chainlit as cl
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.callbacks import CallbackManager
from dotenv import load_dotenv

load_dotenv()

# GLOBAL SETTINGS
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

@cl.on_chat_start
async def start():
    Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])

    # Path for specialized recursive storage
    persist_dir = "./storage_recursive"

    if os.path.exists(persist_dir):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        vector_index = load_index_from_storage(storage_context)
        # Reconstruct retriever with access to the parent docstore
        vector_retriever = vector_index.as_retriever(similarity_top_k=3)
        retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever},
            node_dict=storage_context.docstore.docs, # Fetches Parent nodes via ID
            verbose=True
        )
        await cl.Message(content="Loaded Recursive Index.").send()
    else:
        await cl.Message(content="Building Recursive Index (Small-to-Big)...").send()
        documents = SimpleDirectoryReader("./data").load_data()

        # 1. Define Two Splitters
        parent_splitter = SentenceSplitter(chunk_size=1024) # Big (Synthesis)
        child_splitter = SentenceSplitter(chunk_size=256)   # Small (Retrieval)

        base_nodes = parent_splitter.get_nodes_from_documents(documents)
        all_nodes = []
        
        # 2. Build Parent-Child Links
        for i, parent_node in enumerate(base_nodes):
            # Create sub-nodes from the parent
            child_nodes = child_splitter.get_nodes_from_documents([parent_node])
            
            # Map children to parent using IndexNode (a pointer)
            for child in child_nodes:
                # Content = child text; index_id = parent_node_id
                i_node = IndexNode.from_text_node(child, parent_node.node_id)
                all_nodes.append(i_node)
            
            # Keep parent in the list for reference
            all_nodes.append(parent_node)

        # 3. Build & Persist Index
        vector_index = VectorStoreIndex(all_nodes)
        vector_index.storage_context.persist(persist_dir=persist_dir)
        
        vector_retriever = vector_index.as_retriever(similarity_top_k=3)
        retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever},
            node_dict={n.node_id: n for n in all_nodes},
            verbose=True
        )
        await cl.Message(content="Recursive Index built and saved!").send()

    # 4. Create Query Engine
    query_engine = RetrieverQueryEngine.from_args(retriever, streaming=True)
    cl.user_session.set("query_engine", query_engine)
    await cl.Message(content="Assistant ready with Recursive Retrieval.").send()

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")
    msg = cl.Message(content="", author="Assistant")
    
    response = await query_engine.aquery(message.content)

    async for token in response.async_response_gen():
        await msg.stream_token(token)
    
    # Debug: Check if retrieval returned a large parent context
    if response.source_nodes:
        context_len = len(response.source_nodes[0].node.get_content())
        await msg.stream_token(f"\n\n*(Context size: {context_len} chars)*")
        
    await msg.send()