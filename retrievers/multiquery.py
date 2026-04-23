from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()

# Documents
docs = [
    Document("Gradient descent is used to minimize the loss function in machine learning models.", metadata={"source": "example"}),
    Document("Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models.", metadata={"source": "example"}),
    Document("The method which is used to minimize the loss function in machine learning models is called gradient descent.", metadata={"source": "example"}),
    Document("Gradient descent is a popular optimization algorithm used to minimize the loss function in machine learning models.", metadata={"source": "example"}),
    Document("Neural network uses gradient descent to optimize the weights and biases during training.", metadata={"source": "example"}),
    Document("Support Vector Machines (SVM) is a supervised machine learning algorithm used for classification and regression tasks.", metadata={"source": "example"})
]

# Embeddings
embedding_model = MistralAIEmbeddings(model="mistral-embed")

# Vector DB
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model
)

# Retriever
retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k": 3}
)

# LLM
llm = ChatMistralAI(model="mistral-small-2506")

# Multi Query Retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

# Query
query = "What is gradient descent?"

docs = multi_query_retriever.invoke(query)

print("MultiQuery Results:")
for doc in docs:
    print(doc.page_content)