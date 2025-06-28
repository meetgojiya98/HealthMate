import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "embeddings/faiss_index"

def get_vectorstore():
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(f"FAISS index path '{INDEX_DIR}' does not exist. Build the vectorstore first.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings)
    return vectorstore
