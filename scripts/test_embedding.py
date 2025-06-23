from src.rag.components.embeddings.openai_embedder import OpenAIEmbedder

if __name__ == "__main__":
    texts = [
        "Dies ist ein Testchunk.",
        "Dies ist ein weiterer Beispieltext zur Einbettung."
    ]
    embedder = OpenAIEmbedder()
    embeddings = embedder.embed_texts(texts)

    for i, vec in enumerate(embeddings):
        print(f"Text {i+1} → Embedding Länge: {len(vec)}")
