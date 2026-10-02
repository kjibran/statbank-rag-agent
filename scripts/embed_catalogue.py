from statbank_rag_agent.db import connect
from statbank_rag_agent.embeddings import MODEL_NAME, embed_documents, to_pgvector

with connect() as conn:
    rows = conn.execute(
        "select table_id, search_text from statbank_tables where embedding is null order by table_id"
    ).fetchall()
    print(f"{len(rows)} tables need embeddings")

    if rows:
        vectors = embed_documents([text for _, text in rows])
        updates = [
            {
                "table_id": table_id,
                "embedding": to_pgvector(vector),
                "model": MODEL_NAME,
            }
            for (table_id, _), vector in zip(rows, vectors)
        ]
        with conn.cursor() as cur:
            cur.executemany(
                "update statbank_tables "
                "set embedding = %(embedding)s::vector, embedding_model = %(model)s "
                "where table_id = %(table_id)s",
                updates,
            )
        print(f"Embedded {len(updates)} tables with {MODEL_NAME}")
