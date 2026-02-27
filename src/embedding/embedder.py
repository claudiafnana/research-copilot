from openai import OpenAI
from typing import List


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in resp.data]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]