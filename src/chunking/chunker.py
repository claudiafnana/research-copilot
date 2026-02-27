import tiktoken
from typing import List, Dict, Optional


class TokenChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int, model: str = "gpt-4o-mini"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.encoding_for_model(model)

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        tokens = self.encoder.encode(text)
        chunks = []

        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_str = self.encoder.decode(chunk_tokens)

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_str,
                "token_count": len(chunk_tokens),
                "start_token": start,
                "end_token": min(end, len(tokens)),
                "metadata": metadata or {}
            })

            start += self.chunk_size - self.chunk_overlap
            chunk_id += 1

        return chunks