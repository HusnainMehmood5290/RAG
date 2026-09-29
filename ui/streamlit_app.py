"""Streamlit chat UI wired to the Parent-Document RAG pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow `streamlit run ui/streamlit_app.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from config import ConfigurationError  # noqa: E402
from logging_config import configure_logging  # noqa: E402
from rag.pipeline import answer_question  # noqa: E402

configure_logging()

st.set_page_config(page_title="Parent-Document RAG Chatbot", layout="centered")
st.title("📄 Parent-Document RAG Chatbot")
st.caption("Small chunks for precise search, parent chunks for exact answers.")

try:
    from config import get_settings

    get_settings()  # fail fast with a readable error if .env is incomplete
except ConfigurationError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about your documents...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving & generating..."):
                result = answer_question(user_input)
            answer_text = result["answer"]
            st.markdown(answer_text)
            sources = sorted(
                {
                    d.metadata.get("source", "unknown")
                    for d in result["context"]
                    if d.metadata.get("source")
                }
            )
            if sources:
                st.caption("Sources: " + ", ".join(sources))
        except Exception as exc:
            logging.getLogger("ui").exception("Query failed.")
            answer_text = f"⚠️ Something went wrong: {exc}"
            st.error(answer_text)

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
