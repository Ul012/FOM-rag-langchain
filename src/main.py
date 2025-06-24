# main.py

from src.rag.pipeline.basic_rag_chain import build_rag_chain
from config import config

if __name__ == "__main__":
    chain = build_rag_chain()
    query = "Worum geht es in Artikel 1 der DSGVO?"
    response = chain.invoke(query)
    print("\nAntwort:")
    print(response["result"])
