import sys

from statbank_rag_agent.db import connect

query = sys.argv[1]
with connect() as conn:
    cur = conn.execute(query)
    if cur.description is None:
        print(f"{cur.rowcount} rows affected")
    else:
        print(" | ".join(col.name for col in cur.description))
        for row in cur.fetchall():
            print(" | ".join(str(value) for value in row))
