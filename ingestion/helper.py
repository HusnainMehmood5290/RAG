from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from uuid import uuid4
from dotenv import load_dotenv
from models.model import Models

import  os, pymupdf4llm


# Initialize the models
load_dotenv()
embedding_name=os.getenv("EMBEDDING_NAME")
models = Models()
embeddings = getattr(models,embedding_name,None)


# GET CONSTANTS 
data_folder = "ingestion\\raw_data"
chunk_size = int(os.getenv("CHUNK_SIZE"))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
check_interval = 10


# this method use in load_file
def create_parent_retriever():
    #create local_store
    fs=LocalFileStore(os.getenv("LOCAL_STORE"))
    local_Store=create_kv_docstore(fs)
    
    # create vector store
    vector_store = Chroma(
        collection_name=os.getenv("COLLECTION_NAME"),
        embedding_function=embeddings,
        persist_directory=os.getenv("VECTOR_STORE"),  # Where to save data locally
    )

    child_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # chunk size (characters)
        chunk_overlap=chunk_overlap,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
        # length_function=len
    )

    # Define parent splitter
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=local_Store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    return retriever


# methods
def load_file(file_path):
    # convert pdf file to
    try:
        # Define new directory for storing converted markdown files
        converted_dir = "ingestion/converted"

        # Ensure the directory exists
        os.makedirs(converted_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        converted_md_path = os.path.join(converted_dir, f"{base_name}.md")
        # converted_md_path = os.path.join(converted_dir, "converted.md")
        md_converted=pymupdf4llm.to_markdown(file_path)
        
        with open(converted_md_path,"w",encoding="utf-8") as f:
            f.write(md_converted)
    except Exception as e:
        print(f"ERROR: failed to load file {e}")
        return

    # define parent retriever
    try:
        parent_retriever=create_parent_retriever()
    except Exception as e:
        print(f"Error: during creation of parent retriever {e}")
    
    # Loading data through unstructured
    try:
        loader=UnstructuredMarkdownLoader(converted_md_path)
        document=loader.load()
    except Exception as e:
        print(f"Error: during Loading document {e}")

    # add data to vector store
    try:
        print("adding document...")
        parent_retriever.add_documents(document)
        return parent_retriever
    except Exception as e:
        print(f"Error: during adding data to retriever {e}")
