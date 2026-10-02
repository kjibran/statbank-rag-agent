from functools import lru_cache

from statbank_rag_agent.bm25 import BM25, tokenize
from statbank_rag_agent.db import connect
from statbank_rag_agent.embeddings import embed_query, to_pgvector

RRF_K = 60  # standard constant in Reciprocal Rank Fusion
CANDIDATES = 50  # how many results each method contributes to the fusion


def vector_search(question: str, k: int = 10) -> list[str]:
    """Table ids ranked by meaning: cosine distance between question and search text."""
    with connect(read_only=True) as conn:
        rows = conn.execute(
            "select table_id from statbank_tables where embedding is not null "
            "order by embedding <=> %s::vector limit %s",
            (to_pgvector(embed_query(question)), k),
        ).fetchall()
    return [r[0] for r in rows]


@lru_cache(maxsize=1)
def _bm25_index() -> tuple[list[str], BM25]:
    """Load all search texts once and build the BM25 index in memory."""
    with connect(read_only=True) as conn:
        rows = conn.execute(
            "select table_id, search_text from statbank_tables order by table_id"
        ).fetchall()
    return [r[0] for r in rows], BM25([tokenize(r[1]) for r in rows])


def keyword_search(question: str, k: int = 10) -> list[str]:
    """Table ids ranked by BM25, so rare words like "milk" outweigh common ones like "number"."""
    table_ids, index = _bm25_index()
    scores = index.scores(tokenize(question))
    ranked = sorted(range(len(table_ids)), key=lambda i: scores[i], reverse=True)
    return [table_ids[i] for i in ranked[:k] if scores[i] > 0]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 10) -> list[str]:
    """Combine several rankings: each list adds 1 / (RRF_K + rank) to a table's score."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, table_id in enumerate(ranking, start=1):
            scores[table_id] = scores.get(table_id, 0.0) + 1 / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]


def hybrid_search(question: str, k: int = 10) -> list[str]:
    """Vector and BM25 keyword search fused with Reciprocal Rank Fusion."""
    return reciprocal_rank_fusion(
        [vector_search(question, CANDIDATES), keyword_search(question, CANDIDATES)], k
    )
