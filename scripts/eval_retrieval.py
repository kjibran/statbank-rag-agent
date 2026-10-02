import hashlib
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import mlflow

from statbank_rag_agent.db import connect
from statbank_rag_agent.embeddings import MODEL_NAME
from statbank_rag_agent.evaluation import evaluate, load_questions
from statbank_rag_agent.retrieval import (
    CANDIDATES,
    RRF_K,
    hybrid_search,
    keyword_search,
    vector_search,
    with_year_filter,
)

EVAL_FILE = "eval/retrieval.jsonl"
EXPERIMENT = "table-retrieval"
METHODS = {
    "vector": vector_search,
    "keyword": keyword_search,
    "hybrid": hybrid_search,
    "vector+years": with_year_filter(vector_search),
    "hybrid+years": with_year_filter(hybrid_search),
}

# Usage: eval_retrieval.py                 compare all methods and log them to MLflow
#        eval_retrieval.py hybrid misses   also list the misses of one method
show_misses_for = sys.argv[1] if len(sys.argv) > 2 and sys.argv[2] == "misses" else None


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def file_fingerprint(path: str) -> str:
    """Short hash of the evaluation file, so runs on different test sets are never mixed up."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


questions = load_questions(EVAL_FILE)
types = sorted({q["type"] for q in questions})
results = {name: evaluate(search, questions, k=10) for name, search in METHODS.items()}

print(f"{len(questions)} questions")
print(f"{'method':10} {'hit@1':>6} {'hit@5':>6} {'hit@10':>7} {'MRR':>6}")
for name, r in results.items():
    print(
        f"{name:10} {r['hit@1']:6.2f} {r['hit@5']:6.2f} {r['hit@10']:7.2f} {r['mrr']:6.2f}"
    )


def hit5_by_type(result: dict) -> dict[str, float]:
    by_type = defaultdict(list)
    for p in result["per_question"]:
        by_type[p["type"]].append(p["rank"] is not None and p["rank"] <= 5)
    return {t: sum(h) / len(h) for t, h in by_type.items()}


print("\nhit@5 by question type:")
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

# Log one MLflow run per method
mlflow.set_experiment(EXPERIMENT)
commit, fingerprint = git_commit(), file_fingerprint(EVAL_FILE)
for name, r in results.items():
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"git_commit": commit, "eval_file_sha": fingerprint})
        mlflow.log_params(
            {
                "method": name,
                "embedding_model": MODEL_NAME,
                "rrf_k": RRF_K,
                "candidates_per_method": CANDIDATES,
                "n_questions": len(questions),
            }
        )
        mlflow.log_metrics(
            {
                "hit_at_1": r["hit@1"],
                "hit_at_5": r["hit@5"],
                "hit_at_10": r["hit@10"],
                "mrr": r["mrr"],
            }
        )
        mlflow.log_metrics({f"hit_at_5_{t}": v for t, v in hit5_by_type(r).items()})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "per_question.json"
            out.write_text(json.dumps(r["per_question"], indent=1, ensure_ascii=False))
            mlflow.log_artifact(str(out))
print(f"\nLogged {len(results)} runs to MLflow experiment '{EXPERIMENT}'")

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
