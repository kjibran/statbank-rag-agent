from statbank_rag_agent.catalogue import to_row, upsert_tables
from statbank_rag_agent.statbank import fetch_tables

tables = fetch_tables()
rows = [to_row(t) for t in tables]
print(f"Fetched {len(tables)} tables from Statistics Denmark")
print(f"Upserted {upsert_tables(rows)} rows into statbank_tables")
