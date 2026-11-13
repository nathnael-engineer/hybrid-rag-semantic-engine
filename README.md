# Hybrid Generative RAG & Semantic Retrieval Engine

Multi-stage generative retrieval pipeline combining Qdrant dense vector search, Elasticsearch (BM25) sparse lexical search, Reciprocal Rank Fusion (RRF), BGE-Reranker-Large cross-encoder reranking, and vLLM streaming generation with citation alignment.

## Quick Start

```bash
cp .env.example .env
pip install -r requirements.txt
docker compose up -d qdrant elasticsearch
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
