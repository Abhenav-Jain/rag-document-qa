from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Models
embedding_model = MistralAIEmbeddings(model="mistral-embed")
llm = ChatMistralAI(model="mistral-small-2506")

# Vector Store
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)

# ------------------ QUERY REWRITING ------------------
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user query to make it more clear and specific for document search."),
    ("human", "{query}")
])

def rewrite_query(query):
    response = llm.invoke(rewrite_prompt.invoke({"query": query}))
    return response.content


# ------------------ MAIN PROMPT ------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful AI assistant.

Use ONLY the provided context to answer.

If answer not found, say:
"I could not find the answer in the document."
"""),
    ("human",
     """Chat History:
{history}

Context:
{context}

Question:
{question}
""")
])

# ------------------ MEMORY ------------------
chat_history = []

print("RAG System Created")
print("Press 0 to exit")

while True:
    query = input("Enter your question: ")
    if query == "0":
        break

    # -------- Query Rewrite --------
    better_query = rewrite_query(query)

    # -------- Retrieval --------
    docs = retriever.invoke(better_query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # -------- Memory --------
    history_text = "\n".join(chat_history)

    formatted_prompt = prompt.invoke({
        "context": context,
        "question": query,
        "history": history_text
    })

    # -------- LLM Call --------
    result = llm.invoke(formatted_prompt)
    answer = result.content

    print(f"\nAnswer: {answer}\n")

    # -------- Save Memory --------
    chat_history.append(f"User: {query}")
    chat_history.append(f"AI: {answer}")

    # -------- Logging --------
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(f"\nUser: {query}\n")
        f.write(f"Rewritten: {better_query}\n")
        f.write(f"Answer: {answer}\n")
        f.write("-" * 50)