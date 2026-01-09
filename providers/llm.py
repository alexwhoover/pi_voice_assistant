from gerald import LLM

class GeminiLLM(LLM):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def get_response(self, prompt: str, chat_history: list = None) -> str:
        contents = chat_history.copy()
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents
            )
            return response.text
        except Exception as e:
            if "429" in str(e):
                return "Rate limit hit. Please try again later."
            raise e