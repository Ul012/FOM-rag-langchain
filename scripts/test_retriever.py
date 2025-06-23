from src.rag.components.embeddings.openai_embedder import OpenAIEmbedder
from src.rag.components.vectorstores.faiss_store import FAISSVectorStore
from src.rag.components.retriever.vector_retriever import SimpleRetriever

if __name__ == "__main__":
    query = "Gibt es ein Recht auf Datenschutz?"

    embedder = OpenAIEmbedder().embedder
    store = FAISSVectorStore(embedder)
    store.load_index()  # zuvor gespeicherter Index

    retriever = SimpleRetriever(store.db)
    docs = retriever.retrieve(query)

    for i, doc in enumerate(docs):
        print(f"Dokument {i + 1}:\n{doc.page_content}\n{'-'*40}")
