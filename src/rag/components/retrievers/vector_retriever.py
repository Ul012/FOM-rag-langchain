from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.retrievers import VectorStoreRetriever


class SimpleRetriever:
    def __init__(self, vectorstore: FAISS, top_k: int = 3):
        """
        Wrapper für einen FAISS-basierten Retriever.
        """
        self.retriever: VectorStoreRetriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    def retrieve(self, query: str):
        """
        Führt Retrieval durch.
        """
        return self.retriever.get_relevant_documents(query)
