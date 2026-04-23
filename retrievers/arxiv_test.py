from langchain_community.retrievers import ArxivRetriever

# Create the retriver
retriever = ArxivRetriever(
    load_max_docs = 3,
    load_all_available_meta=True
)

# Query arxiv
docs = retriever.invoke("What is the latest research on deep learning?")

# Print the results
for i,doc in enumerate(docs):
    print("Title: %s" % doc.metadata.get("title", "N/A"))
    print("Authors: %s" % doc.metadata.get("authors", "N/A"))
    print("Document Summary: %s" % doc.page_content[:600])