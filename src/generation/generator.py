from openai import OpenAI
from dotenv import load_dotenv
from src.generation.prompt_strategies import STRATEGIES
from src.retrieval.retriever import RetrievedChunk

load_dotenv()


class OpenAIGenerator:
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model  = model

    def generate_answer(
        self,
        question: str,
        context: str = "",
        chunks: list[RetrievedChunk] = None,
        strategy: str = "default",
    ) -> str:
        if chunks is not None:
            build_prompt = STRATEGIES.get(strategy, STRATEGIES["default"])
            system_prompt, user_message = build_prompt(question, chunks)
        else:
            system_prompt = (
                "You are Research Copilot, an academic assistant.\n"
                "Use ONLY the context below to answer.\n"
                f"CONTEXT:\n{context}"
            )
            user_message = question

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
