from src.retrieval.retriever import RetrievedChunk


def build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c.title} | {c.author} | {c.year}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def strategy_clear_delimiters(question: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
    context = build_context(chunks)
    system = (
        "You are Research Copilot, an academic assistant specialized in agroecology.\n"
        "Answer ONLY using the information inside the <context> tags.\n"
        "If the answer is not there, say: 'I cannot find this in the provided papers.'\n"
        "Write in formal academic tone."
    )
    user = f"<context>\n{context}\n</context>\n\n<question>\n{question}\n</question>"
    return system, user


def strategy_json_output(question: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
    context = build_context(chunks)
    system = (
        "You are Research Copilot, an academic assistant specialized in agroecology.\n"
        "Respond ONLY with valid JSON. No text outside the JSON.\n"
        'Use this schema: {"answer": "...", "confidence": "high|medium|low", '
        '"key_concepts": ["..."], "citations": [{"author":"...","year":"...","title":"..."}], '
        '"limitations": "..."}'
    )
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    return system, user


def strategy_few_shot(question: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
    context = build_context(chunks)
    system = (
        "You are Research Copilot, an academic assistant specialized in agroecology.\n"
        "Answer using ONLY the provided context. Use the examples below as a style guide."
    )
    examples = (
        "EXAMPLE 1:\n"
        "Q: What is the role of biodiversity in agroecological systems?\n"
        "A: Biodiversity plays a foundational role in agroecological systems by enhancing "
        "ecosystem resilience. Diversified farming systems mimic natural ecosystems, enabling "
        "biological pest control and nutrient cycling without synthetic inputs (Altieri, 2002).\n\n"
        "EXAMPLE 2:\n"
        "Q: How does agroecology relate to food sovereignty?\n"
        "A: Agroecology is closely linked to food sovereignty as both prioritize community control "
        "over food systems. Agroecological practices empower smallholder farmers by reducing "
        "reliance on corporate inputs (Rosset & Altieri, 2017).\n\n"
        "---\nNow answer using the provided context in the same academic style:\n"
    )
    user = f"{examples}\nCONTEXT:\n{context}\n\nQUESTION: {question}"
    return system, user


def strategy_chain_of_thought(question: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
    context = build_context(chunks)
    system = (
        "You are Research Copilot, an academic assistant specialized in agroecology.\n"
        "Use ONLY the provided context. Think step by step:\n"
        "STEP 1 - Identify relevant sources.\n"
        "STEP 2 - Extract key claims.\n"
        "STEP 3 - Synthesize into a coherent answer.\n"
        "STEP 4 - Add inline APA citations.\n"
        "FINAL ANSWER: Write the polished academic response."
    )
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    return system, user


STRATEGIES = {
    "default":          strategy_clear_delimiters,
    "clear_delimiters": strategy_clear_delimiters,
    "json_output":      strategy_json_output,
    "few_shot":         strategy_few_shot,
    "chain_of_thought": strategy_chain_of_thought,
}
