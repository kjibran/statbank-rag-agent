import sys

from statbank_rag_agent.db import connect
from statbank_rag_agent.embeddings import embed_query, to_pgvector

question = sys.argv[1]
with connect() as conn:
    rows = conn.execute(
        "select table_id, title, 1 - (embedding <=> %s::vector) as similarity "
        "from statbank_tables order by embedding <=> %s::vector limit 5",
        (to_pgvector(embed_query(question)),) * 2,
    ).fetchall()

print(f"Question: {question}")
for table_id, title, similarity in rows:
    print(f"  {similarity:.3f}  {table_id:10} {title}")
