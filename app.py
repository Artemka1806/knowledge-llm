import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter

DATA_DIR = "./data"
PERSIST_DIR = "./storage"

def build_index():
    docs = SimpleDirectoryReader(DATA_DIR).load_data()
    Settings.llm = OpenAI(model="gpt-5", api_key=OPENAI_API_KEY)
    index = VectorStoreIndex.from_documents(docs, transformations=[SentenceSplitter(chunk_size=512)])
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("Index built and persisted to", PERSIST_DIR)
    return index

def load_or_build_index():
    if Path(PERSIST_DIR).exists() and any(Path(PERSIST_DIR).iterdir()):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print("Loaded index from", PERSIST_DIR)
    else:
        index = build_index()
    return index

def query_index(index, query_text):
    query_engine = index.as_query_engine(streaming=True)
    response_stream = query_engine.query(query_text)
    print("=== RESPONSE ===")
    response_stream.print_response_stream()

if __name__ == "__main__":
    idx = load_or_build_index()
    q = input("Enter your question: ")
    query_index(idx, q)
