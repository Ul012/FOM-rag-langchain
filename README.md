# 📚 RAG-System mit LangChain und SLMs

Dieses Repository enthält ein modulares Retrieval-Augmented Generation (RAG)-System auf Basis von [LangChain](https://www.langchain.com/). Ziel ist die vergleichende Evaluation verschiedener Komponenten wie Chunking-Methoden, Embedding-Modelle, Retrievalstrategien und Vektor-Datenbanken unter Verwendung von Small Language Models (SLMs). Die Anwendung erfolgt am Beispiel der DSGVO.

## 🧱 Projektstruktur

Die Codebasis ist im Sinne guter Softwarepraxis modular aufgebaut:

```
├── .env                    # Konfigurationswerte (nicht versioniert)
├── main.py                # Einstiegspunkt der Anwendung
├── config.py              # Zentrale Konfigurationsklasse (aus .env geladen)
├── src/
│   ├── rag/
│   │   ├── components/
│   │   │   ├── chunking/           # Verschiedene Chunking-Strategien
│   │   │   ├── embeddings/         # SLM-basierte Embedding-Komponenten
│   │   │   ├── vector_store/       # Vektor-Datenbank (z. B. FAISS, in_memory)
│   │   │   ├── retriever/          # Retrieval-Mechanismen
│   │   │   └── slm/                # Sprachmodell-Anbindung (OpenAI, Qwen)
│   │   └── pipeline/
│   │       └── load_data.py        # Laden und Aufteilen der DSGVO-Daten
├── data/
│   └── raw/               # Enthält die Beispieldatei `dsgvo_sample.txt`
├── tests/
│   └── ...                # Modulbezogene Unit-Tests
```

## ✅ Aktueller Stand

Folgende Komponenten sind aktuell implementiert:

- Projektstruktur aufgesetzt mit `.gitignore`, `.env` und `config.py`
- Chunking-Komponente (Baseline & FixedSizeChunker)
- Testdatei `dsgvo_sample.txt` integriert
- Datenlade- und Chunking-Logik über `load_data.py` realisiert
- Erste Tests erfolgreich durchgeführt (Chunk-Validierung über `main.py`)

## 🔜 Nächste Schritte

Die nächsten Entwicklungsschritte umfassen:

1. Implementierung der Embedding-Komponente (OpenAI, ggf. Qwen)
2. Vektor-Datenbank: zunächst InMemoryStore, später FAISS
3. Integration eines einfachen Retrievers (Similarity-Suche)
4. Aufbau einer ersten RAG-Kette
5. Einbindung einer Web-Oberfläche für die Demonstration
6. Evaluierung verschiedener Varianten (z. B. Chunker, Top-k, Similarity)

## ⚙️ Voraussetzungen

- Python 3.10
- Virtuelle Umgebung empfohlen (z. B. `venv`)
- Installation der Basispakete (z. B. via `requirements.txt` oder manuell):
  ```bash
  pip install langchain-core langchain-openai langchain-community langchain-text-splitters python-dotenv
  ```

## 🗂️ Datenbasis

Die DSGVO dient als Grundlage zur Bewertung der Antwortqualität. Aktuell wird eine Auszugsdatei `dsgvo_sample.txt` verwendet. Eine vollständige Verarbeitung der gesamten Verordnung ist perspektivisch vorgesehen.
