import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from src.pipeline.rag_pipeline import RAGPipeline
from src.citation.apa_formatter import deduplicated_citations

QUESTIONS = [
    "What is agroecology and what are its core principles?",
    "How does agroecology contribute to food sovereignty?",
    "What is the relationship between agroecology and biodiversity?",
    "How does agroecology address climate change adaptation?",
    "What are the main socioeconomic benefits of agroecological practices?",
    "How is agroecology different from conventional agriculture?",
    "What role does agroecology play in smallholder food security?",
    "What are the main challenges for scaling up agroecology in Europe?",
    "How does regenerative agriculture relate to agroecology?",
    "What tools are used to measure agroecological performance?",
    "How does agroecology support local food system resilience?",
    "What is the TAPE methodology and how is it used in agroecology?",
    "How does technology integration support agroecological transitions?",
    "What are the key levers for agroecological transition in tropical agriculture?",
    "How does agroecology contribute to climate resilience in Latin America?",
    "What are the decolonial pedagogies used in agroecological learning?",
    "How does policy support influence agroecological transitions?",
    "What is the relationship between agroecology and territorial development?",
    "How do agroecological systems address the socioeconomic issues in farming?",
    "What evidence exists for agroecology's contribution to climate mitigation?",
]

OUTPUT_PATH = "evaluation/results.csv"
STRATEGIES  = ["clear_delimiters", "json_output", "few_shot", "chain_of_thought"]

def main():
    pipeline = RAGPipeline(collection_name="research_copilot_256", chunk_size=256)

    rows = []
    total = len(QUESTIONS) * len(STRATEGIES)
    done  = 0

    for q_idx, question in enumerate(QUESTIONS):
        for strategy in STRATEGIES:
            done += 1
            print(f"[{done}/{total}] Q{q_idx+1} | {strategy} ...")

            result = pipeline.query(question, top_k=5, strategy=strategy)
            citations = deduplicated_citations(result["chunks"])

            rows.append({
                "question_id": q_idx + 1,
                "question":    question,
                "strategy":    strategy,
                "answer":      result["answer"].replace("\n", " ")[:500],
                "citations":   " | ".join(citations[:3]),
                "n_chunks":    len(result["chunks"]),
            })

    os.makedirs("evaluation", exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results saved to {OUTPUT_PATH}")
    print(f"Total rows: {len(rows)} ({len(QUESTIONS)} questions x {len(STRATEGIES)} strategies)")

if __name__ == "__main__":
    main()
