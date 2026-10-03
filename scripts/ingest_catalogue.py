from statbank_rag_agent.catalogue import to_row, upsert_tables
from statbank_rag_agent.db import connect
from statbank_rag_agent.statbank import fetch_tables

# Safety net: never remove tables based on a suspiciously short list, such as during an outage
MIN_EXPECTED_TABLES = 2000

tables = fetch_tables()
print(f"Statbank lists {len(tables)} active tables")
print(f"Upserted {upsert_tables([to_row(t) for t in tables])} tables")

current_ids = [t["id"] for t in tables]
if len(current_ids) < MIN_EXPECTED_TABLES:
    print(
        f"Only {len(current_ids)} tables listed, so no tables are removed as a precaution"
    )
else:
    with connect() as conn:
        removed = conn.execute(
            "delete from statbank_tables where not (table_id = any(%s)) returning table_id",
            (current_ids,),
        ).fetchall()
    names = ", ".join(r[0] for r in removed[:20])
    print(
        f"Removed {len(removed)} tables no longer in Statbank{': ' + names if removed else ''}"
    )
