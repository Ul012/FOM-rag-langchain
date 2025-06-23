# src/rag/scripts/load_dsgvo.py

from pathlib import Path
from rag.components.chunking.fixedsize_chunker import FixedSizeChunker
from rag.components.embeddings.openai_embedder import OpenAIEmbedder
from rag.components.vectorstores.in_memory_vectorstore import InMemoryVectorStore
from rag.config import config


def load_and_index_dsgvo(filepath: Path) -> InMemoryVectorStore:
    """
    Lädt die DSGVO-Datei, erstellt Chunks und speichert sie im VectorStore.
    """
    # Datei einlesen
    text = filepath.read_text(encoding="utf-8")

    # Chunking
    chunker = FixedSizeChunker(chunk_size=200, overlap=40)
    chunks = chunker.chunk(text)
    print(f"✅ DSGVO geladen und in {len(chunks)} Chunks unterteilt.")

    # Embeddings erstellen
    embedder = OpenAIEmbedder()
    embeddings = embedder.embed_docuVERORDNUNG (EU) 2016/679 DES EUROPÄISCHEN PARLAMENTS UND DES RATES

vom 27. April 2016

zum Schutz natürlicher Personen bei der Verarbeitung personenbezogener Daten, zum freien Datenverkehr und zur Aufhebung der Richtlinie 95/46/EG (Datenschutz-Grundverordnung)

Artikel 1
Gegenstand und Ziele

(1) Diese Verordnung enthält Vorschriften zum Schutz natürlicher Personen bei der Verarbeitung personenbezogener Daten und zum freien Verkehr solcher Daten.

(2) Diese Verordnung schützt die Grundrechte und Grundfreiheiten natürlicher Personen und insbesondere deren Recht auf Schutz personenbezogener Daten.

(3) Der freie Verkehr personenbezogener Daten in der Union darf aus Gründen des Schutzes natürlicher Personen bei der Verarbeitung personenbezogener Daten weder eingeschränkt noch verboten werden.

Artikel 2
Sachlicher Anwendungsbereich

(1) Diese Verordnung gilt für die ganz oder teilweise automatisierte Verarbeitung personenbezogener Daten sowie für die nichtautomatisierte Verarbeitung personenbezogener Daten, die in einem Dateisystem gespeichert sind oder gespeichert werden sollen.ments(chunks)

    # Vector Store
    vectorstore = InMemoryVectorStore(similarity=config.vector_store.similarity)
    vectorstore.add_documents(chunks, embeddings)
    print("✅ Embeddings gespeichert im VectorStore.")

    return vectorstore
