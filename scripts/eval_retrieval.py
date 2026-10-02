import sys

from statbank_rag_agent.db import connect
from statbank_rag_agent.evaluation import evaluate, load_questions
from statbank_rag_agent.retrieval import hybrid_search, keyword_search, vector_search

METHODS = {"vector": vector_search, "keyword": keyword_search, "hybrid": hybrid_search}

# Usage: eval_retrieval.py                 compare all methods
#        eval_retrieval.py hybrid misses   also list the misses of one method
show_misses_for = sys.argv[1] if len(sys.argv) > 2 and sys.argv[2] == "misses" else None

questions = load_questions()
results = {name: evaluate(search, questions, k=10) for name, search in METHODS.items()}

print(f"{len(questions)} questions")
print(f"{'method':10} {'hit@1':>6} {'hit@5':>6} {'hit@10':>7} {'MRR':>6}")
for name, r in results.items():
    print(
        f"{name:10} {r['hit@1']:6.2f} {r['hit@5']:6.2f} {r['hit@10']:7.2f} {r['mrr']:6.2f}"
    )

print("\nhit@5 by question type:")
types = sorted({q["type"] for q in questions})
print(f"{'type':16} " + " ".join(f"{name:>8}" for name in METHODS))
for qtype in types:
    row = []
    for name in METHODS:
        hits = [
            p["rank"] is not None and p["rank"] <= 5
            for p in results[name]["per_question"]
            if p["type"] == qtype
        ]
        row.append(f"{sum(hits)}/{len(hits)}")
    print(f"{qtype:16} " + " ".join(f"{cell:>8}" for cell in row))

if show_misses_for:
    with connect(read_only=True) as conn:
        titles = dict(
            conn.execute("select table_id, title from statbank_tables").fetchall()
        )
    print(f"\nMisses for {show_misses_for} (correct table not in the top 5):")
    for p in results[show_misses_for]["per_question"]:
        if p["rank"] is None or p["rank"] > 5:
            found = f"rank {p['rank']}" if p["rank"] else "not in top 10"
            print(f"\n{p['id']} ({found}) {p['question']}")
            for table_id in p["results"][:3]:
                print(f"  got: {table_id} {titles.get(table_id, '?')}")
