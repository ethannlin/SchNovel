'''
    File: generate.py
    Description: This file contains chromadb utility functions for the vector database.
'''

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def get_vector_store(collection_name: str, db_dir: str) -> Chroma:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

    vector_store = Chroma(
        persist_directory=db_dir,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
        embedding_function=embeddings,
        create_collection_if_not_exists=True,
    )

    return vector_store

def clear_vector_store(collection_name: str, db_dir: str) -> None:
    vector_store = get_vector_store(collection_name, db_dir)
    vector_store.reset_collection()

def get_summary(vector_store):
    doc_ids = vector_store.get()['ids']
    metadata = vector_store.get()['metadatas']
    summary = {
        "total_documents": len(doc_ids),
        "sample_document_id": doc_ids[:5],  # Show first 5 document IDs
        "sample_metadata": metadata[:5],   # Show metadata for the first 5 documents
    }
    return summary