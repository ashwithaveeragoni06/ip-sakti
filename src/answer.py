import os
import re
import sys

import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
API_KEY = os.getenv("LLM_API_KEY", "")
if not API_KEY or "paste-your-key" in API_KEY:
    sys.exit("No API key found. Put it in .env as LLM_API_KEY=your-key")

MODEL_NAME = "gemini-3.8-flash"
POOL = 40            # candidates fetched by vector search
TOP_K = 8            # sections finally given to the LLM
MAX_DISTANCE = 0.75  # if even the best vector match is farther than this, refuse

STOPWORDS = {
    "the", "and", "for", "are", "can", "does", "did", "how", "what", "who",
    "when", "where", "which", "that", "this", "with", "from", "have", "has",
    "need", "use", "using", "under", "any", "yes", "not", "you", "your",
    "was", "were", "will", "would", "should", "could", "about", "into",
    "act", "law", "yesterday", "today",
}


def keywords_of(question):
    words = re.findall(r"[a-zA-Z]+", question.lower())
    return [w for w in words if len(w) > 3 and w not in STOPWORDS]


def keyword_score(text, kws):
    text = text.lower()
    hits = sum(1 for w in set(kws) if w in text)
    pairs = sum(1 for a, b in zip(kws, kws[1:]) if f"{a} {b}" in text)
    return 0.04 * hits + 0.12 * pairs


question = " ".join(sys.argv[1:])
if not question:
    sys.exit('Usage: python src\\answer.py "your question"')

# 1. Vector search: fetch a wide pool of candidates
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
collection = chromadb.PersistentClient(path="data/chroma").get_collection("ip_sakti")
res = collection.query(
    query_embeddings=embedder.encode([question]).tolist(),
    n_results=POOL,
)
docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]

# 2. Refuse if nothing in the database is close to the question
best = min(dists)
print(f"\n(best match distance: {best:.3f})")
if best > MAX_DISTANCE:
    print("\nI could not find relevant provisions in my knowledge base for this question.")
    print("Please consult a qualified IP professional.")
    sys.exit()

# 3. Re-rank: vector similarity + keyword bonus
kws = keywords_of(question)
ranked = sorted(
    zip(docs, metas, dists),
    key=lambda x: (1 - x[2]) + keyword_score(x[0], kws),
    reverse=True,
)[:TOP_K]
docs = [r[0] for r in ranked]
metas = [r[1] for r in ranked]

# 4. Build the prompt with numbered sources
sources_text = "\n\n".join(
    f"[{i}] ({m['title']}, Section {m['section']}, Page {m['page']})\n{d}"
    for i, (d, m) in enumerate(zip(docs, metas), start=1)
)

prompt = f"""You are an assistant that explains Indian intellectual property and
biodiversity law for Ayurveda practitioners, researchers and start-ups.

Rules:
- Answer ONLY using the numbered sources below. Do not use outside knowledge.
- Write in simple language, in 4 to 8 sentences.
- After each claim, cite the source like [1] or [2].
- If the sources do not contain the answer, say so clearly and recommend
  consulting a qualified IP professional.
- Do not give legal advice. This is general information.

SOURCES:
{sources_text}

QUESTION: {question}
"""

# 5. Ask the LLM
import time

client = genai.Client(api_key=API_KEY, http_options={"timeout": 60000})
response = None
for attempt in range(1, 6):
    try:
        print(f"Asking Gemini (attempt {attempt}/5), please wait...", flush=True)
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        break
    except Exception as e:
        msg = str(e)
        busy = "503" in msg or "429" in msg or "UNAVAILABLE" in msg or "timed out" in msg.lower()
        if not busy or attempt == 5:
            raise
        wait = 5 * attempt
        print(f"  Gemini is busy. Retrying in {wait} seconds...", flush=True)
        time.sleep(wait)

print("\nANSWER\n------")
print(response.text)

print("\nSOURCES\n-------")
for i, m in enumerate(metas, start=1):
    print(f"[{i}] {m['title']}, Section {m['section']}, Page {m['page']}")

print("\nInformation only, not legal advice.")