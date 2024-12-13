import google.generativeai as genai

from src.constants import GOOGLE_API_KEY


class Gemini:
    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(model_name)

    def __call__(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text.strip()
