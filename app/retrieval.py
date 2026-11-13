from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app.config import settings

@dataclass
class RetrievedPassage:
    chunk_id: str
    text: str
    section_header: str
    score: float

class HybridRetriever:
    def __init__(self) -> None:
        self.embedder = SentenceTransformer(settings.embedding_model)
        self.qdrant = QdrantClient(url=settings.qdrant_url)
        self.es = Elasticsearch(settings.elasticsearch_url)

    def _dense_search(self, query: str, limit: int) -> List[RetrievedPassage]:
        vector = self.embedder.encode(query, normalize_embeddings=True).tolist()
        hits = self.qdrant.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            limit=limit,
        )
        return [
            RetrievedPassage(
                chunk_id=str(h.payload["chunk_id"]),
                text=h.payload["text"],
                section_header=h.payload.get("section_header", "General"),
                score=float(h.score),
            )
            for h in hits
        ]

    def _sparse_search(self, query: str, limit: int) -> List[RetrievedPassage]:
        resp = self.es.search(
            index=settings.elasticsearch_index,
            size=limit,
            query={"match": {"text": {"query": query, "fuzziness": "AUTO"}}},
        )
        return [
            RetrievedPassage(
                chunk_id=hit["_id"],
                text=hit["_source"]["text"],
                section_header=hit["_source"].get("section_header", "General"),
                score=float(hit["_score"]),
            )
            for hit in resp["hits"]["hits"]
        ]

    def reciprocal_rank_fusion(
        self, dense_hits: List[RetrievedPassage], sparse_hits: List[RetrievedPassage]
    ) -> List[RetrievedPassage]:
        fused_scores: Dict[str, float] = {}
        passage_map: Dict[str, RetrievedPassage] = {}
        for ranked_list in (dense_hits, sparse_hits):
            for rank, item in enumerate(ranked_list, start=1):
                fused_scores[item.chunk_id] = fused_scores.get(item.chunk_id, 0.0) + 1.0 / (
                    settings.rrf_k + rank
                )
                passage_map[item.chunk_id] = item
        ordered_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
        return [
            RetrievedPassage(
                chunk_id=passage_map[cid].chunk_id,
                text=passage_map[cid].text,
                section_header=passage_map[cid].section_header,
                score=fused_scores[cid],
            )
            for cid in ordered_ids[: settings.top_k_candidates]
        ]

    def retrieve(self, query: str) -> List[RetrievedPassage]:
        limit = settings.top_k_candidates
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(self._dense_search, query, limit)
            sparse_future = pool.submit(self._sparse_search, query, limit)
            return self.reciprocal_rank_fusion(dense_future.result(), sparse_future.result())
