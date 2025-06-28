import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "embeddings/faiss_index"

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(f"FAISS index directory '{INDEX_DIR}' not found. Run build_vectorstore.py first.")
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectorstore
