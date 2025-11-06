"""
Hugging Face embedding adapter (cloud mode).

Works with the new router endpoint:
https://router.huggingface.co/hf-inference/models/<model>
"""

import os
import time
import requests


class HFEmbedder:
    def __init__(self, model_name=None, api_key=None):
        self.model_name = model_name or os.getenv("HUGGINGFACE_EMBEDDING_MODEL")
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("❌ Missing HUGGINGFACE_API_KEY in .env")

        print(f"✅ HFEmbedder initialized (cloud mode) with model: {self.model_name}")

    # ------------------------------------------------------------------
    def _hf_url(self):
        return f"https://router.huggingface.co/hf-inference/models/{self.model_name}"

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, payload, max_retries=3):
        """POST with retry logic."""
        url = self._hf_url()
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=self._headers(), timeout=60)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                print(f"⚠️ HF API request failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                else:
                    print("❌ HF API permanently failed after 3 attempts.")
                    return None

    # ------------------------------------------------------------------
    def get_embedding(self, text):
        data = self._post({"inputs": text})
        if not data:
            return []
        return data[0] if isinstance(data[0], list) else data

    def get_embeddings(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        data = self._post({"inputs": texts})
        if not data:
            return []
        results = []
        for item in data:
            if isinstance(item, list):
                if len(item) and isinstance(item[0], list):
                    results.append(item[0])
                else:
                    results.append(item)
            else:
                results.append([float(item)])
        print(f"🧩 HFEmbedder: Generated {len(results)} embeddings")
        return results
