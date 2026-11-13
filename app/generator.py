from typing import List
from openai import OpenAI
from app.config import settings
from app.retrieval import RetrievedPassage

SYSTEM_PROMPT = (
    "Answer strictly using the provided context passages and cite supporting "
    "passages inline using [1], [2], etc."
)

class CitationGenerator:
    def __init__(self) -> None:
        self.client = OpenAI(base_url=settings.vllm_base_url, api_key="vllm-local")

    def synthesize(self, query: str, contexts: List[RetrievedPassage]) -> dict:
        numbered = "\n\n".join(
            f"[{idx}] ({ctx.section_header}) {ctx.text}"
            for idx, ctx in enumerate(contexts, start=1)
        )
        completion = self.client.chat.completions.create(
            model=settings.vllm_model,
            temperature=0.0,
            max_tokens=512,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{numbered}\n\nQuestion: {query}"},
            ],
        )
        return {
            "answer": completion.choices[0].message.content,
            "citations": [
                {
                    "ref": idx,
                    "chunk_id": ctx.chunk_id,
                    "section": ctx.section_header,
                    "rerank_score": round(ctx.score, 4),
                }
                for idx, ctx in enumerate(contexts, start=1)
            ],
        }
