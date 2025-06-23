from src.rag.components.slm.openai_slm import OpenAISLM

if __name__ == "__main__":
    slm = OpenAISLM()
    prompt = "Fasse die Grundidee der DSGVO in zwei Sätzen zusammen."
    print(slm.generate(prompt))
