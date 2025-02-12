import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")


list_of_files = [
    "store/local_store/.gitkeep",
    "store/vector_store/.gitkeep",
    "notebooks//trials.ipynb",
    "models/model.py",
    "models/__init__.py",
    "ingestion/raw_data/.gitkeep",  # Stores PDFs, CSVs, etc.
    "ingestion/ingest.py",
    "ingestion/preprocess.py",
    "ingestion/helper.py",
    "ingestion/__init__.py",
    "ingestion/converted/.gitkeep",
    "ui/streamlit_app.py",
    "scripts/clean_store.py",
    ".env",  # Stores environment variables & configuration
    ".gitignore",  # Specifies files to ignore in Git
    "app.py",  # Main application entry point
    "requirements.txt",  # Project dependencies
    "README.md",  # Documentation
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")
