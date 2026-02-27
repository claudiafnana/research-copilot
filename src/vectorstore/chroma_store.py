import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional


class ChromaVectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None

    def create_collection(self, name: str):
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, ids: List[str], documents: List[str],
                      embeddings: List[List[float]], metadatas: List[Dict]):
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query(self, query_embedding: List[float], n_results: int = 5,
              where: Optional[Dict] = None) -> Dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )