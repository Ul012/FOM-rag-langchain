# src/rag/components/retriever/simple_retriever.py

from typing import List
from rag.components.vectorstores.base import VectorStore
from rag.components.embeddings.base import BaseEmbedder
from rag.config import config


class SimpleRetriever:
    """
    Baseline-Retriever: verwendet einen Vektor-Store und Embedding-Modell zur Ähnlichkeitssuche.
    """

    def __init__(self, vectorstore: VectorStore, embedder: BaseEmbedder):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.top_k = config.retrieval.top_k

    def retrieve(self, query: str) -> List[str]:
        """
        Führt eine semantische Suche im Vektorstore aus.

        Args:
            query: Eingabe-Frage

        Returns:
            Liste der ähnlichsten Chunks (Textstrings)
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.vectorstore.search(query_embedding, top_k=self.top_k)
        return [item["text"] for item in results]
