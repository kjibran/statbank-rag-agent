from statbank_rag_agent.db import connect
from statbank_rag_agent.embeddings import MODEL_NAME, embed_documents, to_pgvector

BATCH_SIZE = 100

with connect() as conn:
    rows = conn.execute(
        "select table_id, search_text from statbank_tables where embedding is null order by table_id"
    ).fetchall()
print(f"{len(rows)} tables need embeddings")

for start in range(0, len(rows), BATCH_SIZE):
    batch = rows[start : start + BATCH_SIZE]
    vectors = embed_documents([text for _, text in batch])
    updates = [
        {"table_id": table_id, "embedding": to_pgvector(vector), "model": MODEL_NAME}
        for (table_id, _), vector in zip(batch, vectors)
    ]
    # Save each batch right away, so an interrupted run loses at most one batch
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            "update statbank_tables "
            "set embedding = %(embedding)s::vector, embedding_model = %(model)s "
            "where table_id = %(table_id)s",
            updates,
        )
    print(f"  {start + len(batch)}/{len(rows)} embedded")

print(f"Done with {MODEL_NAME}")
