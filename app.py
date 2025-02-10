from models.model import Models
model=Models()
llm=model.gemini_llm
print(llm.invoke("hi").content)