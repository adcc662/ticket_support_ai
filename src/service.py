from openai import OpenAI
import os

from src.schemas import OpenAIConfig


class TicketClassifierService:
    def __init__(self):
        self.category_model = None
        self.priority_model = None
        self.openai_client = None
        self.openai_config = None
        self.load_models()
        self.setup_openai()

    def setup_openai(self):
        """
        Configuration OpenAI Client
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
            self.openai_config = OpenAIConfig(api_key=api_key)
        else:
            print("OpenAI API Key not set")

    

