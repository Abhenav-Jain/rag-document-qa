from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

Template = ChatPromptTemplate.from_messages([
    ("system", "You are a AI that summarizes the text provided by the user."),
    ("human", "{data}")
])

model = ChatMistralAI(model = "mistral-small-2506")

result = model.invoke(prompt)

print(result.content)

