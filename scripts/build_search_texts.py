from statbank_rag_agent.catalogue import rebuild_search_texts

checked, changed = rebuild_search_texts()
print(f"Checked {checked} tables, rebuilt {changed} search texts")
