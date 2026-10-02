import re
from collections.abc import Callable
from functools import lru_cache

from statbank_rag_agent.bm25 import BM25, tokenize
from statbank_rag_agent.db import connect
from statbank_rag_agent.embeddings import embed_query, to_pgvector

RRF_K = 60  # standard constant in Reciprocal Rank Fusion
CANDIDATES = 50  # how many results each method contributes before fusion or filtering
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")

SearchFunction = Callable[[str, int], list[str]]


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


# --- Year filter ---


def years_in(question: str) -> list[int]:
    """Explicit four-digit years mentioned in a question."""
    return [int(y) for y in YEAR_PATTERN.findall(question)]


def _year_span(
    first_period: str | None, latest_period: str | None
) -> tuple[int, int] | None:
    """Periods like '1901', '2008Q1' or '2021M10' all start with the year."""
    try:
        return int(first_period[:4]), int(latest_period[:4])
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def _table_spans() -> dict[str, tuple[int, int]]:
    with connect(read_only=True) as conn:
        rows = conn.execute(
            "select table_id, first_period, latest_period from statbank_tables"
        ).fetchall()
    spans = {table_id: _year_span(first, latest) for table_id, first, latest in rows}
    return {table_id: span for table_id, span in spans.items() if span is not None}


def filter_by_years(
    candidates: list[str], years: list[int], spans: dict[str, tuple[int, int]]
) -> list[str]:
    """Keep tables whose time range covers every year. Falls back to all candidates if none do."""
    covering = [
        table_id
        for table_id in candidates
        if table_id in spans
        and all(spans[table_id][0] <= y <= spans[table_id][1] for y in years)
    ]
    return covering or candidates


def with_year_filter(search: SearchFunction) -> SearchFunction:
    """Wrap a search function: questions with explicit years only get tables covering them."""

    def filtered(question: str, k: int = 10) -> list[str]:
        years = years_in(question)
        if not years:
            return search(question, k)
        candidates = search(question, CANDIDATES)
        return filter_by_years(candidates, years, _table_spans())[:k]

    return filtered
