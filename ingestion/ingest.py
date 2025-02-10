import os
import time
from dotenv import load_dotenv
# from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
# from models import Models
from models.model import Models

import pymupdf4llm
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Initialize the models
load_dotenv()
embedding_name=os.getenv("EMBEDDING_NAME")
models = Models()
embeddings = hasattr(models,embedding_name,None)


# Define constants
data_folder = "ingestion\\raw_data"
chunk_size = os.getenv("CHUNK_SIZE")
chunk_overlap = os.getenv("CHUNK_OVERLAP")
check_interval = 10


# Chroma vector store
vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("VECTOR_STORE"),  # Where to save data locally
)

# Function to process and store embeddings for a PDF
def ingest_file(file_path):
    if not file_path.lower().endswith('.pdf'):
        print(f"File skipped: {file_path} (Not a PDF)")
        return

    print(f"Processing file: {file_path}")
    # Convert PDF to markdown
    loaded_documents = pymupdf4llm.to_markdown(file_path)

    # Split markdown into headers and chunks
    headers = [
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3"),
        ("####", "Header4"),
        ("#####", "Header5"),
        ("######", "Header6"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(loaded_documents)
    # print(f"\nMD header:\n{md_header_splits}\n\n")

    # Further split into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        is_separator_regex=False,

    )
    document_chunks = text_splitter.split_documents(md_header_splits)
    
    # Generate unique IDs and metadata
    uuids = [str(uuid4()) for _ in range(len(document_chunks))]
    metadata = [
        {"file_name": os.path.basename(file_path), "chunk_index": i}
        for i in range(len(document_chunks))
    ]

    # Add documents to the vector store
    print(f"Adding {len(document_chunks)} chunks to the vector store.")
    vector_store.add_texts(
        texts=[chunk.page_content for chunk in document_chunks],
        metadatas=metadata,
        ids=uuids
    )
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