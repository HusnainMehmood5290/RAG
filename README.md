# Parent-Document RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot using LangChain's
**ParentDocumentRetriever**: small *child* chunks are embedded for precise vector
search, while the large *parent* chunk containing each match is returned to the
LLM — giving answers that quote documents exactly.

## Architecture

```
raw PDFs ──► ingestion worker ──► PDF→Markdown (pymupdf4llm)
                │                 └► split on Markdown headers (H1–H6)
                ▼                 └► child+parent chunking
        Chroma (vector store) + LocalFileStore (doc store)
                ▲
                │  retrieve parents
user query ──► LCEL RAG chain ──► Gemini LLM ──► grounded answer + sources
(CLI: app.py | Web UI: ui/streamlit_app.py)
```

- `config.py` — single, validated source of environment configuration (fails fast with clear messages)
- `models/model.py` — lazy, cached LLM/embedding singletons
- `rag/pipeline.py` — ParentDocumentRetriever + LCEL retrieval→generation chain
- `ingestion/` — content-hash-idempotent PDF watcher (`--once` mode for cron/CI)
- `ui/streamlit_app.py` — chat UI wired to the pipeline, shows sources
- `scripts/clean_store.py` — reset stores (dry-run by default)
- `tests/` — unit tests (config validation, header chunking, ingestion idempotency)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + tests/lint

cp .env.example .env                     # then set GOOGLE_API_KEY

# 1) Ingest documents (drop PDFs into ingestion/raw_data/)
python -m ingestion.ingest               # watch loop
python -m ingestion.ingest --once        # one-shot batch

# 2) Ask questions
python app.py                            # interactive CLI
python app.py --query "What is X?"       # one-shot
streamlit run ui/streamlit_app.py        # web chat UI
```

## Configuration

All settings come from environment variables (see `.env.example`). Only
`GOOGLE_API_KEY` is required; everything else has sensible defaults and is
validated at startup (e.g., `CHUNK_OVERLAP < CHUNK_SIZE`).

## Testing & linting

```bash
pytest                    # unit tests — no API keys or model downloads needed
ruff check .              # lint
ruff format --check .     # formatting
```

## Design notes

- **Idempotent ingestion**: files are indexed under their SHA-256 fingerprint
  and moved to `ingestion/processed/<hash>__<name>`; duplicate re-uploads are
  skipped, failures stay in `raw_data/` for retry.
- **No silent failures**: helpers raise; workers log with full tracebacks.
- **Lazy heavy imports**: embedding models and vector stores load only when
  actually used, keeping CLI startup and tests fast.
- **Grounded prompts**: the system prompt forbids answering without context and
  both UIs display retrieved source documents.
