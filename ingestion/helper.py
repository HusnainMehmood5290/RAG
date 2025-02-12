import os
import pymupdf4llm
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from dotenv import load_dotenv
from models.model import Models
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever

# Load environment variables
load_dotenv()

# Constants
DATA_FOLDER = "ingestion/raw_data"
CONVERTED_DIR = "ingestion/converted"
os.makedirs(CONVERTED_DIR, exist_ok=True)  # Ensure directory exists

# Load embedding model
embedding_name = os.getenv("EMBEDDING_NAME")
models = Models()
embeddings = getattr(models, embedding_name, None)

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


def create_parent_retriever():
    """Creates and returns a ParentDocumentRetriever instance."""
    try:
        local_store = create_kv_docstore(LocalFileStore(os.getenv("LOCAL_STORE")))
        vector_store = Chroma(
            collection_name=os.getenv("COLLECTION_NAME"),
            embedding_function=embeddings,
            persist_directory=os.getenv("VECTOR_STORE"),
        )

        # Define chunking strategy
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

        return ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=local_store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
    except Exception as e:
        print(f"ERROR: Failed to create parent retriever - {e}")
        return None


def convert_pdf_to_markdown(file_path):
    """Converts a PDF file to Markdown and saves it in the converted directory."""
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        converted_md_path = os.path.join(CONVERTED_DIR, f"{base_name}.md")

        md_content = pymupdf4llm.to_markdown(file_path)
        with open(converted_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return converted_md_path
    except Exception as e:
        print(f"ERROR: Failed to convert PDF to Markdown - {e}")
        return None


def load_file(file_path):
    """
    Processes a given PDF file:
    1. Converts it to Markdown.
    2. Loads the Markdown into a retriever.
    3. Adds the document to the vector store.
    """
    converted_md_path = convert_pdf_to_markdown(file_path)
    if not converted_md_path:
        print(f"ERROR: Markdown conversion failed for {file_path}")
        return

    parent_retriever = create_parent_retriever()
    if not parent_retriever:
        print("ERROR: Parent retriever could not be created.")
        return

    try:
        loader = UnstructuredMarkdownLoader(converted_md_path)
        document = loader.load()

        print(f"Adding document {converted_md_path} to retriever...")
        parent_retriever.add_documents(document)

    except Exception as e:
        print(f"ERROR: Processing failed for {converted_md_path} - {e}")
