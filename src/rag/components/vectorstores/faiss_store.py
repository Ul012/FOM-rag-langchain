from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from typing import List
import os
import pickle


class FAISSVectorStore:
    def __init__(self, embedder: Embeddings, persist_path: str = "data/processed/faiss_index"):
        self.embedder = embedder
        self.persist_path = persist_path
        self.db = None

    def build_index(self, texts: List[str], metadata_list: List[dict] = None):
        """
        Erstellt FAISS-Index aus Texten.
        """
        self.db = FAISS.from_texts(texts, self.embedder, metadatas=metadata_list)

    def save_index(self):
        """
        Speichert FAISS-Index auf Platte.
        """
        if self.db is None:
            raise ValueError("Index wurde noch nicht erstellt.")
        self.db.save_local(self.persist_path)

    def load_index(self):
        """
        Lädt FAISS-Index von Platte.
        """
        self.db = FAISS.load_local(self.persist_path, self.embedder)

    def similarity_search(self, query: str, k: int = 3):
        """
        Führt Ähnlichkeitssuche durch.
        """
        if self.db is None:
            raise ValueError("Index ist nicht geladen.")
        return self.db.similarity_search(query, k=k)
