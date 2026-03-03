from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from src.embedding.embedder import OpenAIEmbedder
from src.vectorstore.chroma_store import ChromaVectorStore

load_dotenv()


@dataclass
class RetrievedChunk:
    text:       str
    paper_id:   str
    title:      str
    author:     str
    year:       str
    chunk_size: int
    distance:   float


class Retriever:
    def __init__(self, collection_name: str, top_k: int = 5):
        self.top_k      = top_k
        self.embedder   = OpenAIEmbedder()
        self.store      = ChromaVectorStore()
        self.store.create_collection(collection_name)

    def retrieve(self, query: str, paper_id: Optional[str] = None) -> list[RetrievedChunk]:
        """
        Embed query, search Chroma, return structured RetrievedChunk list.
        If paper_id is provided, filters results to that paper only.
        """
        query_embedding = self.embedder.embed_query(query)

        where = {"paper_id": paper_id} if paper_id else None

        raw = self.store.query(
            query_embedding=query_embedding,
            n_results=self.top_k,
            where=where
        )

        chunks = []
        for text, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0]
        ):
            chunks.append(RetrievedChunk(
                text=text,
                paper_id=meta.get("paper_id", ""),
                title=meta.get("title", "Untitled"),
                author=meta.get("author", "Unknown"),
                year=meta.get("year", "n.d."),
                chunk_size=meta.get("chunk_size", 0),
                distance=dist
            ))

        return chunks
