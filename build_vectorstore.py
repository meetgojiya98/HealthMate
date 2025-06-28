import os
import pandas as pd
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/medquad_clean.csv"
INDEX_DIR = "embeddings/faiss_index"

def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    # Assuming columns are 'question' and 'answer'
    texts = df['question'].tolist()
    answers = df['answer'].tolist()

    # Create Document list for FAISS index
    docs = [Document(page_content=qa, metadata={"answer": ans}) for qa, ans in zip(texts, answers)]
    return docs

def main():
    docs = load_data()
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Creating vectorstore...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    print(f"Saving vectorstore to {INDEX_DIR}...")
    vectorstore.save_local(INDEX_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
