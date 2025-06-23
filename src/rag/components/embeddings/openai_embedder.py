from typing import List
import openai
from langchain.embeddings import OpenAIEmbeddings
from config import config


class OpenAIEmbedder:
    def __init__(self):
        openai.api_key = config.OPENAI_API_KEY
        self.embedder = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL
        )

    def embed_text(self, text: str) -> List[float]:
        """
        Erstellt Embedding für einen einzelnen Text.
        """
        return self.embedder.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Erstellt Embeddings für eine Liste von Texten.
        """
        return self.embedder.embed_documents(texts)
