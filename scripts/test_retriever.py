import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.retriever import Retriever

r = Retriever(collection_name="research_copilot_512", top_k=3)
chunks = r.retrieve("What is agroecology?")

for i, c in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(f"  paper_id:   {c.paper_id}")
    print(f"  title:      {c.title}")
    print(f"  author:     {c.author}")
    print(f"  year:       {c.year}")
    print(f"  chunk_size: {c.chunk_size}")
    print(f"  distance:   {c.distance:.4f}")
    print(f"  text[:80]:  {c.text[:80]}")
    print()
