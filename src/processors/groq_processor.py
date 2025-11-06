import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GroqProcessor:
    """
    Minimal Groq API adapter that provides the same surface used by the project:
    - generate_text(prompt)
    - (embedding methods are stubbed, since embeddings come from Hugging Face)
    """

    def __init__(self, model_name=None, api_key=None, api_url=None):
        """
        Initialize the Groq text generation adapter.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.api_url = api_url or os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not provided. Please set it in your .env file.")

        print(f"✅ GroqProcessor initialized with model: {self.model_name}")

    # ----------------------------------------------------------------------
    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_url(self, kind="chat/completions"):
        """Builds Groq-compatible endpoint URL."""
        base = self.api_url.rstrip("/")
        if f"/{kind}" in base or base.endswith(kind):
            return self.api_url
        return f"{base}/{kind}"

    # ----------------------------------------------------------------------
    def generate_text(self, prompt, max_tokens=512, temperature=0.3):
        """
        Generate a text completion from Groq using OpenAI-compatible schema.
        """
        completions_url = self._build_url(kind="chat/completions")

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a concise and factual assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        try:
            resp = requests.post(completions_url, json=payload, headers=self._headers(), timeout=60)
            if resp.status_code == 400:
                print(f"❌ [Groq API Error 400] Response: {resp.text}")
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and "choices" in data and data["choices"]:
                msg = data["choices"][0].get("message", {})
                if "content" in msg:
                    print(f"🧠 [Groq Debug] Model output: {msg['content'][:200]}...")
                    return msg["content"].strip()

            return data.get("text", "⚠️ No response content returned.")
        except requests.RequestException as e:
            print(f"❌ Error calling Groq API: {e}")
            return "Error generating text from Groq."

    # ----------------------------------------------------------------------
    def get_embedding(self, text):
        """Stub (Hugging Face handles embeddings)."""
        return None

    def get_embeddings(self, texts):
        """Stub (Hugging Face handles embeddings)."""
        return None
