import sys
import time

from rag import answer

sys.stdout.reconfigure(encoding="utf-8")

QUESTIONS = [
    "Is traditional knowledge patentable?",
    "What are not inventions under the Patents Act?",
    "Do I need permission to use a medicinal plant commercially?",
    "How do I register a brand name?",
    "What is a geographical indication?",
    "Can I get a patent for an Ayurvedic herbal formulation?",
]

for q in QUESTIONS:
    print("\n>>", q)
    r = answer(q, "India", use_cache=True, log=print)
    tag = " (already saved)" if r.get("from_cache") else ""
    print("   ->", r["status"], "| confidence:", r["confidence"], "| model:", r["model"], tag)
    time.sleep(2)

print("\nDone. Answers are saved in data\\cache.json")