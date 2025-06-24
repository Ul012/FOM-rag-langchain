# src/interface/streamlit_app.py

import os
import sys

# Projektstruktur für Importe anpassen → bis zur Projektwurzel
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import streamlit as st
from src.rag.pipeline.basic_rag_chain import build_rag_chain

# Streamlit-Setup
st.set_page_config(page_title="DSGVO RAG Demo", layout="centered")
st.title("📚 DSGVO-RAG-Demo mit LangChain & SLM")

# Kette initialisieren (nur einmal)
@st.cache_resource
def get_chain():
    return build_rag_chain()

chain = get_chain()

# Eingabe
user_input = st.text_input("Deine Frage zur DSGVO:")

# Verarbeitung
if user_input:
    with st.spinner("Verarbeite Anfrage..."):
        result = chain.invoke(user_input)
        st.markdown("### 🧠 Antwort:")
        st.write(result["result"])

        # Optional: Quellen anzeigen
        if "source_documents" in result:
            st.markdown("---")
            st.markdown("**🔍 Quellen (Auszüge):**")
            for doc in result["source_documents"]:
                st.markdown(f"• `{doc.page_content[:200].strip()}...`")
