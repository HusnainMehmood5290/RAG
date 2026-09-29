.PHONY: install install-dev test lint format run-cli run-ui ingest clean

install:
pip install -r requirements.txt

install-dev:
pip install -r requirements-dev.txt

test:
pytest

lint:
ruff check .

format:
ruff format .

run-cli:
python app.py

run-ui:
streamlit run ui/streamlit_app.py

ingest:
python -m ingestion.ingest --once

clean:
python scripts/clean_store.py --force
