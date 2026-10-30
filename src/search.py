import sys

import chromadb
from sentence_transformers import SentenceTransformer

question = " ".join(sys.argv[1:]) or "Is traditional knowledge patentable?"

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_collection("ip_sakti")

results = collection.query(
    query_embeddings=model.encode([question]).tolist(),
    n_results=5,
)

print(f"\nQuestion: {question}\n")
for doc, meta, dist in zip(
    results["documents"][0], results["metadatas"][0], results["distances"][0]
):
    print(f"[{meta['title']} | Section {meta['section']} | Page {meta['page']} | distance {dist:.3f}]")
    print(doc[:300], "...\n")