from src.rag.components.embeddings.openai_embedder import OpenAIEmbedder
from src.rag.components.vectorstores.faiss_store import FAISSVectorStore

if __name__ == "__main__":
    texts = [
        "Artikel 1: Jeder Mensch ist gleich.",
        "Artikel 2: Die Freiheit der Meinung ist gewährleistet.",
        "Artikel 3: Datenschutz ist ein Grundrecht."
    ]

    embedder = OpenAIEmbedder().embedder  # Übergib direkt das LangChain-Objekt
    store = FAISSVectorStore(embedder)

    store.build_index(texts)
    store.save_index()

    # Index wieder laden und abfragen
    store.load_index()
    results = store.similarity_search("Was sagt die DSGVO über Meinungsfreiheit?", k=2)

    for r in results:
        print(f"> {r.page_content}")
