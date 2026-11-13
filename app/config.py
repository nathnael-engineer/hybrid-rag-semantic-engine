from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "enterprise_docs"
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "enterprise_bm25"
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-large"
    rrf_k: int = 60
    top_k_candidates: int = 100
    top_n_rerank: int = 7

    class Config:
        env_file = ".env"

settings = Settings()
