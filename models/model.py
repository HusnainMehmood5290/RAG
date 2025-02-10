# from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
class Models:
    def __init__(self):
        # Initialize only the LLM immediately (if needed)
        self._embeddings_HuggingFace = None
        self._gemini_llm=None

    @property
    def gemini_llm(self):
        if self._gemini_llm is None:
            try:
                print("Initializing gemini model")
                self._gemini_llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=api_key)
            except Exception as e:
                print(f"Error initializing  gemini model: {e}")
        return self._gemini_llm
    

    @property
    def embeddings_HuggingFace(self): #best and fastest + suggested by hugingface
        if self._embeddings_HuggingFace is None:
            try:
                print("Initializing HuggingFace Embeddings (all-mpnet-base-v2)...")
                self._embeddings_HuggingFace = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            except Exception as e:
                print(f"Error initializing HuggingFace Embeddings (all-mpnet-base-v2): {e}")
        return self._embeddings_HuggingFace
