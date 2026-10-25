
"""Command-line RAG query against the existing Wikipedia ChromaDB using Groq."""

import os
import sys
from pathlib import Path

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer


# -------------------- CONFIGURATION --------------------

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "db"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

COLLECTION_NAME = "wikipedia"
TOP_K = 3


# -------------------- LOAD KNOWLEDGE BASE --------------------

def load_knowledge_base():
    """Load the existing Wikipedia vector database."""

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set in this PowerShell session."
        )

    if not DB_DIR.exists():
        raise RuntimeError(
            f"Wikipedia database folder not found: {DB_DIR}"
        )

    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"ChromaDB collection '{COLLECTION_NAME}' was not found."
        ) from exc

    if collection.count() == 0:
        raise RuntimeError("The Wikipedia collection is empty.")

    return collection


# -------------------- RETRIEVAL --------------------

def retrieve_documents(question, collection):
    """Retrieve the most relevant passages from Wikipedia."""

    print("Loading embedding model...")

    embedder = SentenceTransformer(EMBED_MODEL)

    question_embedding = embedder.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    result = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    sources = []

    for index, document in enumerate(documents):
        metadata = (
            metadatas[index]
            if index < len(metadatas) and metadatas[index]
            else {}
        )

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        sources.append({
            "text": document,
            "title": metadata.get(
                "title",
                metadata.get("filename", "Wikipedia collection"),
            ),
            "metadata": metadata,
            "distance": distance,
        })

    return sources


# -------------------- GROQ ANSWER GENERATION --------------------

def generate_answer(question, sources):
    """Generate an answer using Groq and retrieved context."""

    if not sources:
        return (
            "I couldn't find relevant information in the Wikipedia "
            "knowledge base. Try rephrasing your question."
        )

    context_parts = []

    for index, source in enumerate(sources, start=1):
        context_parts.append(
            f"[Source {index}: {source['title']}]\n"
            f"{source['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful Retrieval-Augmented Generation (RAG) assistant.

Answer the question using ONLY the retrieved context below.

Rules:
1. Do not invent facts that the context does not support.
2. If the context does not contain the answer, say so clearly.
3. Explain the answer in simple, direct language.
4. Cite supporting information using [Source 1], [Source 2], etc.
5. Do not claim to have read information outside the supplied context.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    errors = []

    for model_name in LLM_MODELS:
        try:
            print(f"Generating answer with {model_name}...")

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.2,
                max_completion_tokens=1024,
            )

            answer_text = response.choices[0].message.content

            if answer_text and answer_text.strip():
                return answer_text.strip()

            errors.append(f"{model_name}: empty response")

        except Exception as exc:
            errors.append(f"{model_name}: {str(exc)[:250]}")
            print(f"Model failed: {model_name}")

    raise RuntimeError(
        "Groq could not generate an answer using the configured models.\n"
        + "\n".join(errors)
    )


# -------------------- COMMAND-LINE INTERFACE --------------------

def answer(question):
    """Retrieve context and print the grounded answer and sources."""

    if not question.strip():
        raise ValueError("Please enter a question.")

    collection = load_knowledge_base()

    print("\nSearching the Wikipedia knowledge base...")
    sources = retrieve_documents(question, collection)

    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(question)

    answer_text = generate_answer(question, sources)

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)
    print(answer_text)

    print("\n" + "=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    if not sources:
        print("No relevant sources were retrieved.")
    else:
        for index, source in enumerate(sources, start=1):
            print(f"\n[Source {index}] {source['title']}")

            if source["distance"] is not None:
                print(f"Cosine distance: {source['distance']:.4f}")

            preview = source["text"].strip()
            print(
                preview[:500] + ("..." if len(preview) > 500 else "")
            )

    return answer_text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            'Usage: python src/query.py "Your question here"\n'
            'Example: python src/query.py "What is machine learning?"'
        )
        raise SystemExit(1)

    question = " ".join(sys.argv[1:])

    try:
        answer(question)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
