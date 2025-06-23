# src/rag/pipeline/load_data.py

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_and_chunk_dsgvo(file_path: Path, chunk_size=1000, chunk_overlap=200) -> list[Document]:
    """
    Lädt die DSGVO-Textdatei und teilt sie in Chunks auf.

    Args:
        file_path: Pfad zur .txt-Datei mit DSGVO-Inhalt.
        chunk_size: Maximale Länge eines Chunks.
        chunk_overlap: Überlappung zwischen Chunks.

    Returns:
        Liste von LangChain-Dokumenten (Chunks).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.create_documents([text])
    return chunks
