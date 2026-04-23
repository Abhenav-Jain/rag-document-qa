# We will learn to use recursive text splitter to split the documents into smaller chunks. 
# This is useful when we have a large document and we want to split it into smaller chunks for better performance.
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Starting...")
data = PyMuPDFLoader(r"D:\Coding\RAG Project\document loaders\book.pdf")
print("Loading...")
docs = data.load()
print("Loaded!")

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=10
)
splits = splitter.split_documents(docs)
print("Chunks:", len(splits))
print(splits[54].page_content)
print(len(splits[54].page_content))