import json
import sys

from statbank_rag_agent.agent import run_agent

question = " ".join(sys.argv[1:]) or "How many people live in Aarhus?"
result = run_agent(question)

print(f"Question: {question}\n")
for i, step in enumerate(result.steps, start=1):
    status = f"error: {step.result['error']}" if "error" in step.result else "ok"
    print(
        f"{i}. {step.tool}({json.dumps(step.arguments, ensure_ascii=False)}) -> {status} [{step.provider}]"
    )

notes = [f"{len(result.providers)} model calls"]
if result.restarts:
    notes.append(f"restarted {result.restarts}x with another provider")
if result.hit_step_limit:
    notes.append("step limit reached")
print(f"\nAnswer ({result.seconds:.1f}s, {', '.join(notes)}):")
print(result.answer)
