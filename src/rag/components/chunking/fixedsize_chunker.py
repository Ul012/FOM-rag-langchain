# src/rag/components/chunking/fixedsize_chunker.py

from typing import List


class FixedSizeChunker:
    """
    Einfacher Chunker, der Text in feste Längen aufteilt.
    Kompatibel zur LangChain-TextSplitter-Konvention durch split_text().
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Teilt den Text in überlappende Chunks auf.

        Args:
            text (str): Vollständiger Eingabetext

        Returns:
            List[str]: Liste von Textchunks
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks
