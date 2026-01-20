import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

def build_recursive_index(data_path="./data", persist_dir="./storage_recursive"):
    """Loads documents and builds a Small-to-Big recursive index."""
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created {data_path} directory. Please add PDFs.")
        return

    print("Loading documents...")
    documents = SimpleDirectoryReader(data_path).load_data()

    # 1. Define Splitters: Parents for context, Children for search
    parent_splitter = SentenceSplitter(chunk_size=1024)
    child_splitter = SentenceSplitter(chunk_size=256)

    base_nodes = parent_splitter.get_nodes_from_documents(documents)
    all_nodes = []
    
    print("Creating parent-child relationships...")
    for parent_node in base_nodes:
        # Create small chunks (Children) from the large chunk (Parent)
        child_nodes = child_splitter.get_nodes_from_documents([parent_node])
        
        for child in child_nodes:
            # Create a pointer node: search the child, retrieve the parent
            i_node = IndexNode.from_text_node(child, parent_node.node_id)
            all_nodes.append(i_node)
        
        all_nodes.append(parent_node)

    # 2. Build and Persist Index to disk
    index = VectorStoreIndex(all_nodes)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"Index successfully saved to {persist_dir}")

if __name__ == "__main__":
    build_recursive_index()