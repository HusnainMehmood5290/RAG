from models.model import Models
import os
from dotenv import load_dotenv
from langgraph.graph import MessagesState,StateGraph
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
        print(llm.invoke(query).content)