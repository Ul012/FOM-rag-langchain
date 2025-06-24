# src/debug/check_pipeline.py

from config import config
from src.rag.pipeline.basic_rag_chain import build_rag_chain
from src.rag.components.chunking.fixedsize_chunker import FixedSizeChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def test_pipeline():
    print("\n[🔧] Starte Systemcheck der RAG-Pipeline...\n")

    # 🔹 1. Chunker-Test
    try:
        text = "Artikel 1: Diese Verordnung enthält Vorschriften zum Schutz natürlicher Personen..."
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split_text(text)
        assert isinstance(chunks, list) and len(chunks) > 0
        print("✅ Chunker-Komponente erfolgreich getestet.")
    except Exception as e:
        print("❌ Fehler in der Chunker-Komponente:", e)

    # 🔹 2. Embedding-Test
    try:
        embeddings = OpenAIEmbeddings(
            model=config.embedding.model,
            api_key=config.embedding.openai_api_key
        )
        _ = embeddings.embed_query("Testeintrag")
        print("✅ Embedding-Komponente erfolgreich getestet.")
    except Exception as e:
        print("❌ Fehler bei der Embedding-Komponente:", e)

    # 🔹 3. VectorStore-Test
    try:
        docs = [Document(page_content=chunk, metadata={}) for chunk in chunks]
        _ = FAISS.from_documents(docs, embeddings)
        print("✅ Vektor-Datenbank erfolgreich aufgebaut.")
    except Exception as e:
        print("❌ Fehler bei der Vektor-Datenbank:", e)

    # 🔹 4. RAG-Kette komplett
    try:
        chain = build_rag_chain()
        test_query = "Worum geht es in Artikel 1 der DSGVO?"
        response = chain.invoke(test_query)
        print("✅ RAG-Kette erfolgreich ausgeführt.")
        print("\nAntwort:", response["result"])
    except Exception as e:
        print("❌ Fehler bei der RAG-Kette:", e)


if __name__ == "__main__":
    test_pipeline()
