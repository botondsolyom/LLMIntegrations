import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from tqdm import tqdm

# load environment variables (.env)
load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DATA_MD = Path("data/book_summaries.md")

client = OpenAI()


def parse_md(md_text: str):
    """Parse the markdown summaries file into (title, summary) items."""
    blocks = re.split(r"\n## Title:\s*", md_text)
    items = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        lines = b.splitlines()
        title = lines[0].strip()
        summary = " ".join(lines[1:]).strip()
        items.append({"title": title, "summary": summary})
    return items


def main():
    if not DATA_MD.exists():
        raise FileNotFoundError(f"Missing file: {DATA_MD}")

    text = DATA_MD.read_text(encoding="utf-8")
    items = parse_md(text)

    # Initialize Chroma
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = chroma.get_or_create_collection(name="book_summaries")

    ids, docs, metas, embeds = [], [], [], []
    for i, it in enumerate(tqdm(items, desc="Embedding books")):
        content = f"Title: {it['title']}\n{it['summary']}"
        emb = client.embeddings.create(model=EMBED_MODEL, input=content).data[0].embedding
        ids.append(f"doc-{i}")
        docs.append(content)
        metas.append({"title": it["title"]})
        embeds.append(emb)

    coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)
    print(f"✅ Uploaded {len(ids)} documents into Chroma at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
