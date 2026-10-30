import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from rag import answer  # noqa: E402

st.set_page_config(page_title="IP-SAKTI Sahayak", page_icon="🌿", layout="wide")

EXAMPLES = [
    "Is traditional knowledge patentable?",
    "What are not inventions under the Patents Act?",
    "Do I need permission to use a medicinal plant commercially?",
    "How do I register a brand name?",
    "What is a geographical indication?",
    "Can I get a patent for an Ayurvedic herbal formulation?",
]


def set_question(q):
    st.session_state["question"] = q


st.session_state.setdefault("question", "")

with st.sidebar:
    st.header("Settings")
    jurisdiction = st.radio("Jurisdiction", ["India", "International"])
    use_cache = st.checkbox("Use saved answers (faster)", value=True)
    st.markdown("**Example questions**")
    for ex in EXAMPLES:
        st.button(ex, on_click=set_question, args=(ex,), use_container_width=True)

st.title("🌿 IP-SAKTI Sahayak")
st.caption("Ayurveda IP & Regulatory Assistant: answers grounded in Indian statutes, with citations")

question = st.text_area("Ask your question", key="question", height=100,
                        placeholder="e.g. Can I patent my Ayurvedic hair oil?")
ask = st.button("Ask", type="primary")

if ask:
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Searching the Acts and preparing an answer..."):
            result = answer(question, jurisdiction, use_cache=use_cache)
        status = result["status"]

        if status == "no_docs":
            st.warning(f"No {jurisdiction} documents are in the knowledge base yet. "
                       "International sources (PCT, Madrid, TRIPS, Nagoya) are planned for the next phase.")
        elif status == "refused":
            st.error("I could not find relevant provisions in my knowledge base for this question. "
                     "Please consult a qualified IP professional.")
        elif status == "llm_failed":
            st.warning("The AI service is unavailable right now, so here are the most relevant "
                       "provisions found in the knowledge base:")
        else:
            st.subheader("Answer")
            st.markdown(result["answer"])

        if status in ("ok", "llm_failed"):
            st.markdown(f"**Confidence:** {result['confidence']} "
                        f"(retrieval similarity {result['best_distance']})")
            st.subheader("Sources")
            for s in result["sources"]:
                label = f"[{s['n']}] {s['title']}, Section {s['section']}, Page {s['page']}"
                with st.expander(label):
                    st.write(s["excerpt"])

        st.warning("⚠ Information only, not legal advice.")
        st.caption("Need more help? A human IP facilitator referral is planned for the next phase.")