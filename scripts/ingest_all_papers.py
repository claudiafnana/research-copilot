import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from src.pipeline.rag_pipeline import RAGPipeline

MANIFEST_PATH = "papers/papers_manifest.json"
PAPERS_DIR    = "papers"

# Two chunk configurations required by the rubric
CONFIGS = [
    {"collection": "research_copilot_256",  "chunk_size": 256,  "overlap": 25},
    {"collection": "research_copilot_1024", "chunk_size": 1024, "overlap": 100},
]

def main():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"Found {len(papers)} papers in manifest.\n")

    for config in CONFIGS:
        print(f"{'='*60}")
        print(f"Indexing into: {config['collection']}  (chunk_size={config['chunk_size']})")
        print(f"{'='*60}")

        pipeline = RAGPipeline(
            collection_name=config["collection"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["overlap"],
        )

        total_chunks = 0
        for paper in papers:
            pdf_path = os.path.join(PAPERS_DIR, paper["filename"])

            if not os.path.exists(pdf_path):
                print(f"  [MISSING] {pdf_path} — skipping.")
                continue

            override = {
                "title":   paper.get("title", ""),
                "author":  paper.get("author", ""),
                "year":    paper.get("year", "n.d."),
                "journal": paper.get("journal", ""),
            }

            print(f"  Ingesting {paper['paper_id']}: {paper['title'][:50]}...")
            n = pipeline.ingest_paper(
                pdf_path=pdf_path,
                paper_id=paper["paper_id"],
                override_metadata=override,
            )
            if n > 0:
                print(f"    -> {n} chunks stored.")
            total_chunks += n

        print(f"\nTotal new chunks stored in {config['collection']}: {total_chunks}\n")

if __name__ == "__main__":
    main()
