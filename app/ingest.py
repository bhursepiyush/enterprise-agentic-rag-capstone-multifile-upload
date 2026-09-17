import os
import pandas as pd
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

# MiniLM for local semantic search embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./data/chroma")
collection = chroma_client.get_or_create_collection(name="documents")

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        text = df.to_string()
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
        text = df.to_string()
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def index_document(file_path: str, doc_name: str):
    text = extract_text(file_path)
    chunks = chunk_text(text)
    if not chunks:
        return 0
    
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{doc_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_name} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    return len(chunks)