import os
import pymupdf4llm
# from langchain_community.document_loaders import UnstructuredMarkdownLoader
from dotenv import load_dotenv
from models.model import Models
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import MarkdownHeaderTextSplitter

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
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))


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
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)

        return ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=local_store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
    except Exception as e:
        print(f"ERROR: Failed to create parent retriever - {e}")
        return None


def document_splitter(file):
    md_content = pymupdf4llm.to_markdown(file)  #convert to markdwon
    headers_to_split_on = [ #this thing store the meta data base on header
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 3"),
        ("#####", "Header 3"),
        ("######", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on) #split on the base of define header
    markdwon_splitted_document = markdown_splitter.split_text(md_content)
    return markdwon_splitted_document #return that spilited data



def load_file(file_path):
    """
    Processes a given PDF file:
    1. Converts it to Markdown.
    2. Loads the Markdown into a retriever.
    3. Adds the document to the vector store.
    """
    try:
        header_to_split=document_splitter(file=file_path)
    except Exception as e:
        print(f"ERROR: while splitting through markdwonHeaderTextSplitter")
    parent_retriever = create_parent_retriever()
    if not parent_retriever:
        print("ERROR: Parent retriever could not be created.")
        return

    try:
        print(f"Adding document {file_path} to retriever...")
        parent_retriever.add_documents(header_to_split)

    except Exception as e:
        print(f"ERROR: Processing failed for {file_path} - {e}")

