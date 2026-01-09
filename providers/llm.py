from gerald import LLM
from google import genai
from google.genai import types
from datetime import datetime

class GeminiLLM(LLM):
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def get_response(self, prompt: str, chat_history: list = None) -> str:
        contents = chat_history.copy()
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