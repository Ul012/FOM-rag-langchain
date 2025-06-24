# tools/convert_pdf_to_txt.py

import pdfplumber
import os

# Dynamisch relativer Pfad zur DSGVO-Datei
base_dir = os.path.dirname(os.path.dirname(__file__))  # geht von tools/ zurück auf Projektbasis
pdf_path = os.path.join(base_dir, "data", "raw", "dsgvo.pdf")
txt_path = os.path.join(base_dir, "data", "raw", "dsgvo.txt")

def convert_pdf_to_txt(pdf_path: str, txt_output_path: str):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"  # Absatztrennung zwischen Seiten

    os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write(full_text.strip())

# Ausführung
if __name__ == "__main__":
    convert_pdf_to_txt(pdf_path=pdf_path, txt_output_path=txt_path)
