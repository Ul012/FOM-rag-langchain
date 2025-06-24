import os
from dataclasses import dataclass, field
from typing import Literal
from dotenv import load_dotenv

# .env-Datei laden
load_dotenv()


@dataclass
class SLMConfig:
    model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    hf_token: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")


@dataclass
class EmbeddingConfig:
    model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")


@dataclass
class VectorStoreConfig:
    store_type: Literal["faiss", "chroma", "in_memory"] = os.getenv("VECTOR_STORE_TYPE", "faiss")
    similarity: Literal["cosine", "dot_product", "euclidean"] = os.getenv("VECTOR_STORE_SIMILARITY", "cosine")


@dataclass
class RetrievalConfig:
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    threshold: float = float(os.getenv("RETRIEVAL_THRESHOLD", "0.3"))


@dataclass
class WebConfig:
    port: int = int(os.getenv("FLASK_PORT", "5000"))
    debug: bool = os.getenv("FLASK_DEBUG", "0") == "1"


@dataclass
class AppConfig:
    slm: SLMConfig = field(default_factory=SLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    web: WebConfig = field(default_factory=WebConfig)
    data_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "../data"))

    def validate(self):
        if not self.slm.openai_api_key:
            raise ValueError("Fehlender OpenAI API-Key. Bitte setze OPENAI_API_KEY in der .env-Datei.")
        return True


# Globale Instanz
config = AppConfig()
