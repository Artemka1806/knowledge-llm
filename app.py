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

def get_indexed_files(index):
    return set([doc.id_ for doc in list(index.docstore.docs.values())])

def build_or_update_index():
    docs = SimpleDirectoryReader(DATA_DIR).load_data()
    Settings.llm = OpenAI(model="gpt-5", api_key=OPENAI_API_KEY)
    if Path(PERSIST_DIR).exists() and any(Path(PERSIST_DIR).iterdir()):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        indexed_ids = get_indexed_files(index)
        new_docs = [doc for doc in docs if doc.doc_id not in indexed_ids]
        if new_docs:
            for doc in new_docs:
                index.insert(doc)
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            print(f"Додано {len(new_docs)} нових документів до індексу.")
        else:
            print("Нових документів не знайдено.")
    else:
        index = VectorStoreIndex.from_documents(docs, transformations=[SentenceSplitter(chunk_size=512)])
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("Побудовано новий індекс.")
    return index

def query_index(index, query_text):
    query_engine = index.as_query_engine(streaming=True)
    response_stream = query_engine.query(query_text)
    print("=== RESPONSE ===")
    response_stream.print_response_stream()

if __name__ == "__main__":
    idx = build_or_update_index()
    while True:
        q = input("\nВаше питання: ")
        query_index(idx, q)
