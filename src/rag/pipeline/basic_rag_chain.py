# src/rag/pipeline/basic_rag_chain.py

import os
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import config
from src.rag.components.chunking.fixedsize_chunker import FixedSizeChunker


def build_rag_chain():
    # Absoluten Pfad zur DSGVO-Datei berechnen
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    file_path = os.path.join(base_dir, "data", "raw", "dsgvo.txt")

    print(f"[INFO] DSGVO-Datei wird geladen von: {file_path}")

    # 1. Daten laden und aufteilen
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    chunker = FixedSizeChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.split_text(text)
    docs = [Document(page_content=chunk, metadata={}) for chunk in chunks]

    # 2. Embeddings generieren
    embeddings = OpenAIEmbeddings(
        model=config.embedding.model,
        api_key=config.embedding.openai_api_key
    )

    # 3. Vektor-Datenbank aufbauen
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 4. Retriever bauen
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.retrieval.top_k}
    )

    # 5. Sprachmodell (SLM)
    slm = ChatOpenAI(
        model_name=config.slm.model,
        temperature=config.slm.temperature,
        api_key=config.slm.openai_api_key
    )

    # 6. RetrievalQA-Kette
    qa_chain = RetrievalQA.from_chain_type(
        llm=slm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
