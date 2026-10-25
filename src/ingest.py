"""Ingest pipeline: stream Wikipedia, chunk, embed, persist to ChromaDB.

Run once before querying. Re-running requires deleting ``db/`` first,
since ``collection.add`` rejects duplicate IDs.
"""

import os
from itertools import islice

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "db")
COLLECTION_NAME = "wikipedia"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NUM_ARTICLES = 500
CHUNK_SIZE = 175
CHUNK_OVERLAP = 50


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split ``text`` into overlapping word windows.

    Returns a list of strings, each containing up to ``size`` whitespace-split
    words, advancing by ``size - overlap`` words between chunks.
    """
    if overlap >= size:
        raise ValueError(
            f"chunk overlap ({overlap}) must be smaller than chunk size ({size}); "
            "otherwise the window never advances"
        )
    words = text.split()
    if not words:
        return []
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunk = words[start : start + size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + size >= len(words):
            break
    return chunks


def main():
    """Stream ``NUM_ARTICLES`` articles, embed each chunk, persist to Chroma."""
    os.makedirs(DB_DIR, exist_ok=True)

    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Connecting to ChromaDB at", os.path.abspath(DB_DIR))
    client = chromadb.PersistentClient(path=os.path.abspath(DB_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print("Streaming wikipedia (20231101.simple)...")
    dataset = load_dataset(
        "wikimedia/wikipedia", "20231101.simple", split="train", streaming=True
    )

    total_chunks = 0
    for i, article in enumerate(islice(dataset, NUM_ARTICLES)):
        title = article.get("title", f"article_{i}")
        text = article.get("text", "") or ""
        chunks = chunk_text(text)
        if not chunks:
            continue

        embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
        ids = [f"{i}-{j}" for j in range(len(chunks))]
        metadatas = [{"title": title, "article_index": i, "chunk_index": j} for j in range(len(chunks))]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} articles, {total_chunks} chunks total")

    print(f"Done. Ingested {total_chunks} chunks from {NUM_ARTICLES} articles.")


if __name__ == "__main__":
    main()
