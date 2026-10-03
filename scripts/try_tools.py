import json

from statbank_rag_agent.tools import (
    describe_table,
    find_values,
    get_data,
    search_tables,
)


def show(title, result):
    print(f"\n=== {title} ===")
    print(json.dumps(result, indent=1, ensure_ascii=False)[:1500])


show(
    "search_tables('population by municipality')",
    search_tables("population by municipality"),
)
show("describe_table('FOLK1A')", describe_table("FOLK1A"))
show(
    "find_values('FOLK1A', 'OMRÅDE', 'Århus')", find_values("FOLK1A", "OMRÅDE", "Århus")
)
show(
    "get_data, Aarhus, latest quarter",
    get_data("FOLK1A", {"OMRÅDE": ["751"], "Tid": ["2026K3"]}),
)
show(
    "get_data without time (should explain the problem)",
    get_data("FOLK1A", {"OMRÅDE": ["751"]}),
)
