import csv
import re
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

RAW_DIR = Path("data/raw")
CSV_PATH = Path("data/documents.csv")
DB_PATH = "data/chroma"

# A line like "3. What are not inventions" starts a new section
SECTION_RE = re.compile(r"^(\d+[A-Z]{0,2})\.\s+\S")
# Footnote lines like "1. Subs. by Act 33 of 2021..." are NOT sections
FOOTNOTE_RE = re.compile(
    r"^\d+\.?\s*(Subs\.|Ins\.|Omitted|Rep\.|Added|The words|Clause|Sub-section|Sec\.)"
)
INDIACODE_RE = re.compile(r"IndiaCode\s*\d*")

MIN_CHARS = 60      # ignore tiny pieces
CHUNK_WORDS = 120   # small enough for the embedding model to read fully
OVERLAP_WORDS = 20  # repeat a few words so ideas aren't cut in half


def read_pdf(path):
    """Return a list of (page_number, text)."""
    reader = PdfReader(str(path))
    return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]


def split_into_sections(pages):
    """Cut the text wherever a new numbered section begins."""
    sections = []
    current = {"section": "Preamble", "page": 1, "lines": []}

    for page_no, text in pages:
        for line in text.split("\n"):
            line = INDIACODE_RE.sub("", line).strip()
            if not line or FOOTNOTE_RE.match(line):
                continue
            match = SECTION_RE.match(line)
            if match:
                sections.append(current)
                current = {"section": match.group(1), "page": page_no, "lines": []}
            current["lines"].append(line)
    sections.append(current)

    result = []
    for s in sections:
        text = " ".join(s["lines"])
        if len(text) >= MIN_CHARS:
            result.append({"section": s["section"], "page": s["page"], "text": text})
    return result


def make_pieces(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    pieces = []
    start = 0
    while start < len(words):
        pieces.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return pieces


def main():
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        docs = list(csv.DictReader(f))

    print("Loading embedding model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    model.max_seq_length = 256

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection("ip_sakti")   # start fresh each run
    except Exception:
        pass
    collection = client.create_collection("ip_sakti", metadata={"hnsw:space": "cosine"})

    for doc in docs:
        path = RAW_DIR / doc["filename"]
        if not path.exists():
            print(f"SKIP (file not found): {doc['filename']}")
            continue

        pages = read_pdf(path)
        total_chars = sum(len(t) for _, t in pages)
        print(f"\n{doc['filename']}: {len(pages)} pages, {total_chars} characters")
        if total_chars < 1000:
            print("  WARNING: almost no text found. This PDF may be a scan. Skipping.")
            continue

        sections = split_into_sections(pages)

        texts, metas, ids = [], [], []
        n = 0
        for s in sections:
            # short heading, e.g. "What are not inventions."
            heading = re.sub(r"^\d+[A-Z]{0,2}\.\s*", "", s["text"].split("—")[0])[:100]
            prefix = f"{doc['title']}, Section {s['section']} ({heading}). "
            for piece in make_pieces(s["text"]):
                n += 1
                texts.append(prefix + piece)
                metas.append({
                    "title": doc["title"],
                    "section": s["section"],
                    "page": s["page"],
                    "jurisdiction": doc["jurisdiction"],
                    "topic": doc["topic"],
                    "source_url": doc["source_url"],
                })
                ids.append(f"{Path(doc['filename']).stem}_{n}")

        print(f"  {len(sections)} sections, {len(texts)} chunks. Embedding...")
        embeddings = model.encode(texts, show_progress_bar=True).tolist()
        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)

    print("\nDone. Total chunks stored:", collection.count())


if __name__ == "__main__":
    main()