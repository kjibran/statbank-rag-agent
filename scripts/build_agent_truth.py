import json
import time
from pathlib import Path

from statbank_rag_agent.agent_eval import load_questions, true_value

truths, problems = {}, 0
for q in load_questions():
    if q.get("answerable", True) is False:
        print(f"{q['id']}  (should be refused)")
        continue
    try:
        truths[q["id"]] = true_value(q["truth"])
        print(f"{q['id']}  {truths[q['id']]:>14,.0f}  {q['question']}")
    except Exception as exc:  # noqa: BLE001 - report every problem, then fix the definitions
        problems += 1
        print(f"{q['id']}  PROBLEM: {exc}")
    time.sleep(0.3)

Path("eval/agent_truth.json").write_text(json.dumps(truths, indent=1))
print(
    f"\nSaved {len(truths)} correct answers to eval/agent_truth.json, {problems} problems"
)
