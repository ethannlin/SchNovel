'''
    File: generate.py
    Description: This file creates a vector database from the RAG datasets using chromadb.
'''

from chroma_utils import get_vector_store, clear_vector_store, get_summary
from langchain_core.documents import Document
import json
from tqdm import tqdm

# Path to RAG vector db data
FILENAME = ""

# Desired name of the database
COLLECTION_NAME = ""

# Directory to the category's RAG db
DB_DIR = ""

def generate_embeddings(filename: str, collection_name: str) -> None:
    clear_vector_store(collection_name, DB_DIR)
    vector_store = get_vector_store(collection_name, DB_DIR)

    # add documents to the collection
    data = []
    with open(filename, "r") as file:
        for line in file:
            data.append(json.loads(line))

    documents = []
    batch_size = 100
    for i, document in tqdm(enumerate(data)):
        doc = Document(
            page_content=document['abstract'],
            metadata={
                "title": document['title'],
                "authors": document['authors'],
                "categories": document['categories'],
                "published_date": document['versions'][0]['created'],
                "url": f"https://arxiv.org/abs/{document['id']}",
            },
            id=document['id'],
        )
        documents.append(doc)

        if (i + 1) % batch_size == 0:
            vector_store.add_documents(documents)
            documents = []
    if documents:
        vector_store.add_documents(documents)

if __name__ == "__main__":
    generate_embeddings(filename=FILENAME, collection_name=COLLECTION_NAME)

    vector_store = get_vector_store(COLLECTION_NAME, DB_DIR)
    summary = get_summary(vector_store)
    print(summary)