import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
RESULTS_DIR = Path("eval/results")
EXPERIMENT = "agent-evaluation"
PAUSE_SECONDS = 60  # lets Groq's per-minute token budget refill fully between questions
LOCAL_TZ = ZoneInfo("Europe/Copenhagen")

# Usage: eval_agent.py              all questions, logged to MLflow
#        eval_agent.py a01,u01      only the listed questions (saved locally, not logged)
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


# Fail fast: check the MLflow connection before spending 20 minutes on questions
if only is None:
    try:
        mlflow.set_experiment(EXPERIMENT)
    except Exception as exc:  # noqa: BLE001 - any connection problem should stop the run now
        sys.exit(
            f"MLflow is not reachable, so the run was not started: {exc}\n"
            "Did you run with: uv run --env-file .env python scripts/eval_agent.py ?"
        )

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
        tokens=result.tokens,
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

    passed = record["correct"] if answerable else record["refused"]
    verdict = (
        ("CORRECT" if record["correct"] else "WRONG")
        if answerable
        else ("REFUSED" if record["refused"] else "ANSWERED")
    )
    cited = "cited" if record["cited"] else "no source"
    print(
        f"{q['id']}  {verdict:8} {cited:9} {record['steps']} steps  "
        f"{record['tokens']:>6,} tokens  {record['seconds']:5.1f}s  {result.tables_used}"
    )
    if not passed:
        expected = (
            f"expected {record['truth']:,.0f}"
            if answerable and record.get("truth")
            else "expected a refusal"
        )
        print(f"     {expected}. Answer: {result.answer[:300]}")

# Always save locally first, so a failure at the end never loses the results
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results_file = RESULTS_DIR / f"agent_{datetime.now(LOCAL_TZ):%Y%m%d_%H%M}.json"
results_file.write_text(json.dumps(records, indent=1, ensure_ascii=False))

answerable_records = [r for r in records if r["answerable"]]
unanswerable_records = [r for r in records if not r["answerable"]]
completed = [r for r in records if "error" not in r]


def share(items: list, key: str) -> float:
    return sum(1 for r in items if r.get(key)) / len(items) if items else 0.0


def average(key: str) -> float:
    return sum(r[key] for r in completed) / len(completed) if completed else 0.0


metrics = {
    "accuracy": share(answerable_records, "correct"),
    "citation_rate": share(answerable_records, "cited"),
    "refusal_rate": share(unanswerable_records, "refused"),
    "error_rate": 1 - len(completed) / len(records) if records else 0.0,
    "avg_steps": average("steps"),
    "avg_model_calls": average("model_calls"),
    "avg_tokens": average("tokens"),
    "avg_seconds": average("seconds"),
}

print(
    f"\n{len(answerable_records)} answerable, {len(unanswerable_records)} unanswerable questions"
)
for name, value in metrics.items():
    print(f"  {name:16} {value:,.2f}")
print(f"\nFull results with traces saved to {results_file}")

if only:
    print("Partial run, not logged to MLflow.")
    sys.exit()

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
    mlflow.log_artifact(str(results_file))
print("Logged to MLflow experiment 'agent-evaluation'")
