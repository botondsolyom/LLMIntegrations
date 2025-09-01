# smart_librarian/rag.py
import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# --- robust paths / env ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

CHROMA_DIR = os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# --- clients ---
_openai = OpenAI()  # needs OPENAI_API_KEY in .env
_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _chroma.get_or_create_collection("book_summaries")

def _embed(text: str) -> List[float]:
    """Create an OpenAI embedding for a single string."""
    return _openai.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def search_books(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Natural-language query -> top-k results from Chroma."""
    if not query.strip():
        return []

    q_emb = _embed(query.strip())
    res = _collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    out: List[Dict[str, Any]] = []
    if res and res.get("ids") and res["ids"][0]:
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({
                "title": meta.get("title", "Unknown"),
                "document": doc,
                "distance": float(dist),
            })
    return out

if __name__ == "__main__":
    print(f"Using CHROMA_DIR: {CHROMA_DIR}")
    try:
        print("Collection size:", _collection.count())
    except Exception as e:
        print("Count failed:", e)

    tests = [
        "I want a book about friendship and magic",
        "What do you recommend if I love war stories?",
        "freedom and social control",
    ]
    for q in tests:
        print(f"\nQ: {q}")
        hits = search_books(q, k=3)
        if not hits:
            print("  (no results)")
        for i, h in enumerate(hits, 1):
            print(f"  {i}. {h['title']}  (dist: {h['distance']:.3f})")
