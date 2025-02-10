import os
import time
from ingestion.helper import load_file

# Constants
DATA_FOLDER = "ingestion/raw_data"
CHECK_INTERVAL = 10  # Time interval for checking new PDFs


def ingest_file(file_path):
    """Processes and stores embeddings for a PDF file."""
    filename = os.path.basename(file_path)

    # Skip already processed PDFs (files starting with '_')
    if filename.startswith("_"):
        print(f"Skipping (Already Processed): {file_path}")
        return

    if not filename.lower().endswith(".pdf"):
        print(f"Skipping (Not a PDF): {file_path}")
        return

    print(f"Processing: {file_path}")
    try:
        load_file(file_path)
        mark_file_as_processed(file_path)
    except Exception as e:
        print(f"ERROR: {file_path} could not be processed - {e}")


def mark_file_as_processed(file_path):
    """Renames the processed file to prevent reprocessing."""
    try:
        new_file_path = os.path.join(DATA_FOLDER, f"_{os.path.basename(file_path)}")
        os.rename(file_path, new_file_path)
        print(f"Marked as processed: {new_file_path}")
    except Exception as e:
        print(f"ERROR: Failed to rename {file_path} - {e}")


def scan_and_ingest():
    """Scans the raw data folder and processes unprocessed PDFs."""
    pdf_files = [
        f for f in os.listdir(DATA_FOLDER) 
        if f.lower().endswith(".pdf") and not f.startswith("_")
    ]

    for filename in pdf_files:
        file_path = os.path.join(DATA_FOLDER, filename)
        ingest_file(file_path)


def main_loop():
    """Continuously checks for new PDF files in the raw data folder."""
    print("Watching for new files... Press Ctrl+C to stop.")
    try:
        while True:
            scan_and_ingest()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("\nProcess interrupted. Exiting gracefully...")


if __name__ == "__main__":
    main_loop()
