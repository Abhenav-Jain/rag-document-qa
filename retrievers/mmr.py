# MMR Retriever
# Steps to retrieve relevant documents using MMRRetriever:
# 1. Define your documents and create their embeddings using a language model (e.g., MistralAIEmbeddings).
# 2. Create a vector store (e.g., Chroma) and populate it with documents and their embeddings.
# 3. Initialize the MMRRetriever with the vector store and a language model for re-ranking.
# 4. Use the MMRRetriever to query for relevant documents based on a user query.
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from tiktoken import model
from dotenv import load_dotenv

load_dotenv()

# Step 1: Define your documents and create their embeddings
docs = [
    Document("Gradient descent is used to minimize the loss function in machine learning models.", metadata={"source": "example"}),
    Document("Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models.", metadata={"source": "example"}),
    Document("The method which is used to minimize the loss function in machine learning models is called gradient descent.", metadata={"source": "example"}),
    Document("Gradient descent is a popular optimization algorithm used to minimize the loss function in machine learning models.", metadata={"source": "example"}),
    Document("Neural network uses gradient descent to optimize the weights and biases during training.", metadata={"source": "example"}),
    Document("Support Vector Machines (SVM) is a supervised machine learning algorithm used for classification and regression tasks.", metadata={"source": "example"})
]

embedding_model = MistralAIEmbeddings(model = "mistral-embed")

# Step 2: Create a vector store and populate it with documents and their embeddings
vector_store = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model
)

similarity_retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 3}
)

print("Similarity Retriever Results:")
similarity_results = similarity_retriever.invoke("What is gradient descent?")

for doc in similarity_results:
    print("{}".format(doc.page_content))

# Step 3: Initialize the MMRRetriever with the vector store and a language model for re-ranking
mmr_retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k": 3}
)

print("\nMMR Retriever Results:")
mmr_docs = mmr_retriever.invoke("What is gradient descent?")

for doc in mmr_docs:
    print("{}".format(doc.page_content))