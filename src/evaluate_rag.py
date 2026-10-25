"""Batch evaluation: run a fixed set of probe questions through the RAG pipeline.

Reuses ``retrieve`` and ``build_prompt`` from ``query.py`` so the eval path
is identical to the interactive path. Output is written to
``data/eval_results.txt`` (overwritten each run).
"""

import os

import chromadb
from sentence_transformers import SentenceTransformer
from mlx_lm import load, generate

from query import build_prompt, retrieve, EMBED_MODEL, LLM_MODEL, COLLECTION_NAME, MAX_TOKENS

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "db")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_results.txt")

TEST_QUESTIONS = [
    "Who was Albert Einstein?",
    "What is the capital of France?",
    "What is photosynthesis?",
    "When did World War II end?",
    "What is the Great Wall of China?",
]


def main():
    """Run every probe in ``TEST_QUESTIONS`` and write a consolidated report."""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    embedder = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=os.path.abspath(DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    print("Loading Phi-2 via mlx-lm...")
    model, tokenizer = load(LLM_MODEL)

    lines = []
    for q in TEST_QUESTIONS:
        print(f"\n--- Q: {q} ---")
        contexts = retrieve(q, embedder, collection)
        prompt = build_prompt(q, contexts)
        response = generate(
            model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False
        )

        lines.append(f"Question: {q}\n")
        lines.append("Retrieved chunks:\n")
        for i, (doc, meta) in enumerate(contexts, 1):
            title = meta.get("title", "unknown")
            lines.append(f"  [{i}] {title}\n")
            lines.append(f"      {doc[:400]}{'...' if len(doc) > 400 else ''}\n")
        lines.append(f"Answer: {response}\n")
        lines.append("=" * 80 + "\n")

        print(response)

    with open(RESULTS_PATH, "w") as f:
        f.writelines(lines)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
