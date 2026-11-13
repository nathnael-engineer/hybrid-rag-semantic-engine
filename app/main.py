import time
from fastapi import FastAPI
from pydantic import BaseModel
from app.generator import CitationGenerator
from app.reranker import NeuralReranker
from app.retrieval import HybridRetriever

app = FastAPI(title="Hybrid Generative RAG & Semantic Retrieval Engine")
retriever = HybridRetriever()
reranker = NeuralReranker()
generator = CitationGenerator()

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/query")
def query_pipeline(payload: QueryRequest) -> dict:
    t0 = time.perf_counter()
    candidates = retriever.retrieve(payload.query)
    t_ret = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    top_passages = reranker.rerank(payload.query, candidates)
    t_rerank = (time.perf_counter() - t1) * 1000
    result = generator.synthesize(payload.query, top_passages)
    result["telemetry_ms"] = {
        "retrieval": round(t_ret, 2),
        "rerank": round(t_rerank, 2),
        "total": round((time.perf_counter() - t0) * 1000, 2),
    }
    return result
