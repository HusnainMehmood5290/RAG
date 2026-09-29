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


@st.cache_resource(show_spinner=False)
def init_stores():
    """Validate config and warm up stores/models once per server process.

    ``cache_resource`` keeps heavy objects (embedding model, Chroma client)
    across reruns so page refreshes stay fast; failures still surface in the
    UI because exceptions are not cached.
    """
    from config import get_settings
    from ingestion.helper import get_ingest_retriever

    settings = get_settings()  # fail fast with a readable error if .env is incomplete
    get_ingest_retriever(settings)  # build retriever + load embedding model once
    return settings


try:
    with st.status("Initializing retrieval backend...", expanded=True) as status:
        settings = init_stores()
        st.write(f"Collection: `{settings.collection_name}` • k={settings.retrieval_k}")
        status.update(label="Backend ready", state="complete", collapsed=True)
except ConfigurationError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()
except Exception:  # model download / store errors should be visible too
    logging.getLogger("ui").exception("Backend initialization failed.")
    # SECURITY: full details (paths, stack) go only to the server log; the
    # browser sees a generic message so internals are never leaked.
    st.error(
        "Initialization failed. Check the server logs for details "
        "(set LOG_LEVEL=DEBUG for more information)."
    )
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
            history = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]
            with st.spinner("Retrieving & generating..."):
                result = answer_question(user_input, history=history)
            st.markdown(result.answer)
            sources = sorted(
                {
                    d.metadata.get("source", "unknown")
                    for d in result.context
                    if d.metadata.get("source")
                }
            )
            if sources:
                st.caption("Sources: " + ", ".join(sources))
            if result.truncated:
                st.caption("ℹ️ Some documents were omitted to fit the context budget.")
            answer_text = result.answer
        except Exception:
            # SECURITY: log the real error server-side, show a generic message.
            # Raw exception text can contain filesystem paths, collection
            # names, or provider error payloads that must not reach the client.
            logging.getLogger("ui").exception("Query failed.")
            answer_text = (
                "⚠️ Something went wrong while answering. Please try again; "
                "the server logs have the details."
            )
            st.error(answer_text)

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
