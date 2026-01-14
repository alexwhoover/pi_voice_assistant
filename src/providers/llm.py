from google import genai
from google.genai import types
from datetime import datetime

from src.interfaces import LLM

class GeminiLLM(LLM):
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def convert_to_provider_format(self, chat_history: list) -> list:
        """Gemini uses its own format, so just return as-is."""
        return chat_history.copy() if chat_history else []

    def convert_from_provider_format(self, provider_history: list) -> list:
        """Gemini uses its own format, so just return as-is."""
        return provider_history.copy() if provider_history else []

    def get_response(self, prompt: str, chat_history: list = None) -> str:
        contents = self.convert_to_provider_format(chat_history or [])
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=f"Current time: {now}",
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )
            return response.text
        except Exception as e:
            if "429" in str(e):
                return "Rate limit hit. Please try again later."
            raise e