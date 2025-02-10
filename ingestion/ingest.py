import os
import time
from .helper import load_file


# Define constants
data_folder = "ingestion\\raw_data"
check_interval = 10


# Function to process and store embeddings for a PDF
def ingest_file(file_path):
    if not file_path.lower().endswith('.pdf'):
        print(f"File skipped: {file_path} (Not a PDF)")
        return

    print(f"Processing file: {file_path}")
    retriever=load_file(file_path)
    print(retriever.get_relevant_documents("Promotion Approval Authority"))
    print(f"File {file_path} processed and stored in the vector store.")

# Main loop
def main_loop():
    while True:
        for filename in os.listdir(data_folder):
            if not filename.startswith("_"):
                file_path = os.path.join(data_folder, filename)
                ingest_file(file_path)
                new_filename = "_" + filename
                new_file_path = os.path.join(data_folder, new_filename)
                os.rename(file_path, new_file_path)
        time.sleep(check_interval)  # Check the folder every 10 seconds

# Run the main loop
if __name__ == "__main__":
    main_loop()