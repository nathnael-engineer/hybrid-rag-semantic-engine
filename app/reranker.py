from typing import List
from sentence_transformers import CrossEncoder
from app.config import settings
from app.retrieval import RetrievedPassage

class NeuralReranker:
    def __init__(self) -> None:
        self.model = CrossEncoder(settings.reranker_model, max_length=512)

    def rerank(
        self, query: str, candidates: List[RetrievedPassage], top_n: int | None = None
    ) -> List[RetrievedPassage]:
        if not candidates:
            return []
        k = top_n or settings.top_n_rerank
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)
        scored = [
            RetrievedPassage(
                chunk_id=c.chunk_id,
                text=c.text,
                section_header=c.section_header,
                score=float(score),
            )
            for c, score in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]
