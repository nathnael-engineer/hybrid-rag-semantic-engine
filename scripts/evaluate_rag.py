import math
from typing import List

def ndcg_at_k(relevances: List[int], k: int = 10) -> float:
    rels = relevances[:k]
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rels))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0

if __name__ == "__main__":
    print(f"Benchmark NDCG@10: {ndcg_at_k([3, 2, 3, 0, 1, 2, 1, 0, 1, 0], k=10):.4f}")
