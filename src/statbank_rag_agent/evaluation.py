import json
from collections.abc import Callable
from pathlib import Path

SearchFunction = Callable[[str, int], list[str]]


def load_questions(path: str = "eval/retrieval.jsonl") -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def first_relevant_rank(results: list[str], relevant: list[str]) -> int | None:
    """1-based rank of the first correct table, or None if none was returned."""
    for rank, table_id in enumerate(results, start=1):
        if table_id in relevant:
            return rank
    return None


def evaluate(search: SearchFunction, questions: list[dict], k: int = 10) -> dict:
    """Run every question through `search` and compute hit rates and MRR."""
    per_question = []
    for q in questions:
        results = search(q["question"], k)
        per_question.append(
            {
                **q,
                "results": results,
                "rank": first_relevant_rank(results, q["relevant"]),
            }
        )

    n = len(per_question)
    ranks = [p["rank"] for p in per_question]
    return {
        "hit@1": sum(1 for r in ranks if r is not None and r <= 1) / n,
        "hit@5": sum(1 for r in ranks if r is not None and r <= 5) / n,
        "hit@10": sum(1 for r in ranks if r is not None and r <= 10) / n,
        "mrr": sum(1 / r for r in ranks if r is not None) / n,
        "per_question": per_question,
    }
