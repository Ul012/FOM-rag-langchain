from src.rag.components.chunking.fixed_size_chunker import FixedSizeChunker

if __name__ == "__main__":
    text = "Dies ist ein Testtext. " * 50
    chunker = FixedSizeChunker(chunk_size=100, overlap=20)
    chunks = chunker.split(text)

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:\n{chunk}\n{'-' * 40}")
