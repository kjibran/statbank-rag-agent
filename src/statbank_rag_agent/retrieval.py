from statbank_rag_agent.db import connect
from statbank_rag_agent.embeddings import embed_query, to_pgvector


def vector_search(question: str, k: int = 10) -> list[str]:
    """Table ids ranked by meaning: cosine distance between question and search text."""
    with connect(read_only=True) as conn:
        rows = conn.execute(
            "select table_id from statbank_tables where embedding is not null "
            "order by embedding <=> %s::vector limit %s",
            (to_pgvector(embed_query(question)), k),
        ).fetchall()
    return [r[0] for r in rows]
