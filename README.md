# 📚 RAG-System mit LangChain und SLMs

Dieses Repository enthält ein modulares Retrieval-Augmented Generation (RAG)-System auf Basis von [LangChain](https://www.langchain.com/). Ziel ist die vergleichende Evaluation verschiedener Komponenten wie Chunking-Methoden, Embedding-Modelle, Retrievalstrategien und Vektor-Datenbanken unter Verwendung von Small Language Models (SLMs). Die Anwendung erfolgt am Beispiel der DSGVO.

## 🧱 Projektstruktur

Die Codebasis ist im Sinne guter Softwarepraxis modular aufgebaut:

```
├── .env                         # Konfigurationswerte (nicht versioniert)
├── requirements.txt            # Paketabhängigkeiten
├── mkdocs.yml                  # Dokumentationskonfiguration (optional)
├── main.py                     # Testskript zur Inferenz
├── config.py                   # Globale Konfiguration (lädt .env)
├── src/
│   ├── interface/              # Streamlit-App
│   │   └── streamlit_app.py    # Weboberfläche
│   ├── debug/                  # Debug- und Testskripte
│   └── rag/
│       ├── components/
│       │   ├── chunking/       # Chunking-Strategien
│       │   └── ...             # Weitere Komponenten (Embeddings, SLM, Retriever)
│       └── pipeline/           # Aufbau der RAG-Kette
├── data/
│   └── raw/                    # Beispieldatei `dsgvo.txt`
└── tools/                      # Hilfsskripte (z. B. PDF-Konvertierung)
```

## ✅ Aktueller Stand

Folgende Komponenten sind aktuell implementiert:

- Projektstruktur mit `.gitignore`, `.env` und `config.py`
- Chunking-Komponente (Baseline & FixedSizeChunker)
- Beispieltext `dsgvo_sample.txt` integriert
- Pipeline zum Laden und Chunken der Daten (`load_data.py`)
- Erste Tests erfolgreich durchgeführt (`check_pipeline.py`)
- Streamlit-Oberfläche lauffähig (`streamlit_app.py`)

## 🔜 Nächste Schritte

- Erweiterung um verschiedene Embedding-Modelle (z. B. OpenAI, Qwen)
- Vergleich FAISS vs. InMemoryStore
- Integration alternativer Retrievalstrategien
- Evaluierung unterschiedlicher RAG-Konfigurationen

## ⚙️ Voraussetzungen

- Python 3.10
- Virtuelle Umgebung empfohlen (z. B. `venv`)
- Installation der Basispakete (z. B. via `requirements.txt` oder manuell):
  ```bash
  pip install langchain-core langchain-openai langchain-community langchain-text-splitters python-dotenv
  ```

## 🗂️ Datenbasis

Die DSGVO dient als Grundlage zur Bewertung der Antwortqualität. Aktuell wird eine Auszugsdatei `dsgvo_sample.txt` verwendet. Eine vollständige Verarbeitung der gesamten Verordnung ist perspektivisch vorgesehen.
