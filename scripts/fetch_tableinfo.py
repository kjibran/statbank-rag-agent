import time

from statbank_rag_agent.catalogue import save_tableinfo, tables_needing_tableinfo
from statbank_rag_agent.statbank import fetch_tableinfo

PAUSE_SECONDS = 0.25  # be polite to Statistics Denmark's servers

todo = tables_needing_tableinfo()
print(f"{len(todo)} tables need metadata")

for i, table_id in enumerate(todo, start=1):
    save_tableinfo(table_id, fetch_tableinfo(table_id))
    if i % 100 == 0 or i == len(todo):
        print(f"  {i}/{len(todo)} done")
    time.sleep(PAUSE_SECONDS)
