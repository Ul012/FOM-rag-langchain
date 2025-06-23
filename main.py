# src/main.py

from config import config
from rag.components.chunking.fixedsize_chunker import FixedSizeChunker
from rag.components.slm.openai_slm import OpenAISLM


def main():
    # Konfiguration prüfen
    config.validate()

    # Dummy-Text
    text = "Die Datenschutz-Grundverordnung (DSGVO) ist eine Verordnung der Europäischen Union. Sie regelt die Verarbeitung personenbezogener Daten durch private Unternehmen und öffentliche Stellen."

    # Chunker verwenden
    chunker = FixedSizeChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk(text)
    print("\n📌 Erste Chunks:")
    for i, c in enumerate(chunks):
        print(f"Chunk {i+1}: {c}")

    # SLM aufrufen
    slm = OpenAISLM()
    answer = slm.generate("Was ist die DSGVO
