import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlflow

from statbank_rag_agent import llm
from statbank_rag_agent.agent import MAX_STEPS, run_agent
from statbank_rag_agent.agent_eval import (
    TOLERANCE,
    is_cited,
    is_correct,
    is_refusal,
    load_questions,
)
from statbank_rag_agent.config import settings

QUESTIONS_FILE = "eval/agent.jsonl"
TRUTH_FILE = "eval/agent_truth.json"
EXPERIMENT = "agent-evaluation"
PAUSE_SECONDS = 45  # lets Groq's per-minute token budget refill between questions

# Usage: eval_agent.py              all questions
#        eval_agent.py a01,u01      only the listed questions (no MLflow logging)
only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def fingerprint(*paths: str) -> str:
    """Short hash of the question and truth files, so runs on different test sets are never mixed up."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()[:12]


questions = [
    q for q in load_questions(QUESTIONS_FILE) if only is None or q["id"] in only
]
truths = json.loads(Path(TRUTH_FILE).read_text())
records = []

for i, q in enumerate(questions):
    if i > 0:
        time.sleep(PAUSE_SECONDS)
    answerable = q.get("answerable", True)
    record = {"id": q["id"], "question": q["question"], "answerable": answerable}
    try:
        result = run_agent(q["question"])
    except llm.LLMError as exc:
        record.update(error=str(exc), correct=False, cited=False, refused=False)
        records.append(record)
        print(f"{q['id']}  ERROR  {exc}")
        continue

    record.update(
        answer=result.answer,
        steps=len(result.steps),
        model_calls=len(result.providers),
        seconds=round(result.seconds, 1),
        restarts=result.restarts,
        hit_step_limit=result.hit_step_limit,
        tables_used=result.tables_used,
        cited=is_cited(result.answer),
        trace=[
            {
                "tool": s.tool,
                "arguments": s.arguments,
                "result": json.dumps(s.result, ensure_ascii=False)[:800],
            }
            for s in result.steps
        ],
    )
    if answerable:
        record["truth"] = truths.get(q["id"])
        record["correct"] = record["truth"] is not None and is_correct(
            result.answer, record["truth"]
        )
        record["refused"] = False
    else:
        record["correct"] = False
        record["refused"] = is_refusal(result.answer)
    records.append(record)

    verdict = (
        ("CORRECT" if record["correct"] else "WRONG")
        if answerable
        else ("REFUSED" if record["refused"] else "ANSWERED")
    )
    cited = "cited" if record["cited"] else "no source"
    print(
        f"{q['id']}  {verdict:8} {cited:9} {record['steps']} steps  {record['seconds']:5.1f}s  {result.tables_used}"
    )

answerable_records = [r for r in records if r["answerable"]]
unanswerable_records = [r for r in records if not r["answerable"]]
completed = [r for r in records if "error" not in r]


def share(items: list, key: str) -> float:
    return sum(1 for r in items if r.get(key)) / len(items) if items else 0.0


metrics = {
    "accuracy": share(answerable_records, "correct"),
    "citation_rate": share(answerable_records, "cited"),
    "refusal_rate": share(unanswerable_records, "refused"),
    "error_rate": 1 - len(completed) / len(records) if records else 0.0,
    "avg_steps": sum(r["steps"] for r in completed) / len(completed)
    if completed
    else 0.0,
    "avg_model_calls": sum(r["model_calls"] for r in completed) / len(completed)
    if completed
    else 0.0,
    "avg_seconds": sum(r["seconds"] for r in completed) / len(completed)
    if completed
    else 0.0,
}

print(
    f"\n{len(answerable_records)} answerable, {len(unanswerable_records)} unanswerable questions"
)
for name, value in metrics.items():
    print(f"  {name:16} {value:.2f}")

if only:
    print("\nPartial run, not logged to MLflow.")
    sys.exit()

mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name="agent"):
    mlflow.set_tags(
        {
            "git_commit": git_commit(),
            "eval_files_sha": fingerprint(QUESTIONS_FILE, TRUTH_FILE),
        }
    )
    mlflow.log_params(
        {
            "groq_model": settings.groq_model,
            "gemini_model": settings.gemini_model,
            "max_steps": MAX_STEPS,
            "tolerance": TOLERANCE,
            "n_questions": len(records),
        }
    )
    mlflow.log_metrics(metrics)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "agent_results.json"
        out.write_text(json.dumps(records, indent=1, ensure_ascii=False))
        mlflow.log_artifact(str(out))
print("\nLogged to MLflow experiment 'agent-evaluation'")
