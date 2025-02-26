from models.model import Models
import os
from dotenv import load_dotenv
from ingestion.helper import create_parent_retriever
# from langgraph.graph import MessagesState,StateGraph
# Load environment variables
load_dotenv()

# Get the model property name from .env
model_name = os.getenv("MODEL_NAME")
# print(model_name)
# Initialize the model class
model = Models()

# Dynamically access the model property
llm = getattr(model, model_name, None)


# print(llm.invoke("hi").content)  # If it's an LLM, invoke it
query=""
while True:
    query=input("Enter query (press q to quit): ")
    if query=="q":
        break
    else:
        retriever=create_parent_retriever()
        retrieved=retriever.invoke(query)
        print(retrieved)
        # print(llm.invokhie(query).content)