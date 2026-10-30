import chromadb

c = chromadb.PersistentClient(path="data/chroma").get_collection("ip_sakti")
r = c.get(where={"title": "The Patents Act 1970"}, include=["metadatas"])
secs = sorted(
    {m["section"] for m in r["metadatas"]},
    key=lambda s: (int("".join(ch for ch in s if ch.isdigit()) or 0), s),
)
print(len(secs), "distinct sections")
print(secs)