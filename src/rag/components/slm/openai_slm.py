from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from config import config


class OpenAISLM:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )

    def generate(self, prompt: str) -> str:
        """
        Führt eine einfache Prompt-Abfrage durch.
        """
        response = self.llm([HumanMessage(content=prompt)])
        return response.content
