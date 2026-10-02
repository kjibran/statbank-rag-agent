from collections import defaultdict

from statbank_rag_agent.db import connect
from statbank_rag_agent.evaluation import evaluate, load_questions
from statbank_rag_agent.retrieval import vector_search

questions = load_questions()
result = evaluate(vector_search, questions, k=10)

print(f"Vector search on {len(questions)} questions")
print(f"  hit@1  {result['hit@1']:.2f}")
print(f"  hit@5  {result['hit@5']:.2f}")
print(f"  hit@10 {result['hit@10']:.2f}")
print(f"  MRR    {result['mrr']:.2f}")

by_type = defaultdict(list)
for p in result["per_question"]:
    by_type[p["type"]].append(p["rank"] is not None and p["rank"] <= 5)
print("\nhit@5 by question type:")
for qtype, hits in sorted(by_type.items()):
    print(f"  {qtype:16} {sum(hits)}/{len(hits)}")

with connect(read_only=True) as conn:
    titles = dict(conn.execute("select table_id, title from statbank_tables").fetchall())

print("\nMisses (correct table not in the top 5):")
for p in result["per_question"]:
    if p["rank"] is None or p["rank"] > 5:
        found = f"rank {p['rank']}" if p["rank"] else "not in top 10"
        print(f"\n{p['id']} ({found}) {p['question']}")
        print(f"  expected: {', '.join(f'{t} {titles.get(t, "?")}' for t in p['relevant'])}")
        for table_id in p["results"][:3]:
            print(f"  got:      {table_id} {titles.get(table_id, '?')}")
