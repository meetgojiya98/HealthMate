import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

DATA_FILE = "medquad_clean.csv"
INDEX_DIR = "embeddings/faiss_index"

def load_data():
    df = pd.read_csv(DATA_FILE)
    texts = df["answer"].tolist()
    metadatas = [{"source": url} for url in df.get("source", ["No source"] * len(df))]
    docs = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas)]
    return docs

def main():
    print("Loading data...")
    docs = load_data()
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Creating vectorstore...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"Saving vectorstore to {INDEX_DIR}...")
    vectorstore.save_local(INDEX_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
