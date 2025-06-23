from typing import List
from .base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """
    Einfacher Chunker mit fester Zeichenlänge und optionalem Overlap.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks
