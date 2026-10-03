import json
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

import gradio as gr
import spaces

from statbank_rag_agent.agent import AgentResult, run_agent
from statbank_rag_agent.llm import LLMError

DAILY_LIMIT = (
    100  # questions per day for the whole app, to protect the free model quotas
)
MAX_QUESTION_CHARS = 300
LOCAL_TZ = ZoneInfo("Europe/Copenhagen")

EXAMPLES = [
    "How many people live in Aarhus?",
    "What was the population of Denmark in 1950?",
    "How many people born in Poland lived in Denmark in 2020?",
    "How many divorces were there in Denmark in 2023?",
    "How many people living in Denmark were born on 29 February?",
]

DESCRIPTION = """Ask a question about Denmark. An AI agent searches the tables of
[Statistics Denmark](https://www.statbank.dk), fetches the actual numbers and cites its source.
Every answer shows the steps the agent took to find it.

This is a demo running on free AI models, so answers can be slow, and occasionally wrong.
Code: [github.com/kjibran/statbank-rag-agent](https://github.com/kjibran/statbank-rag-agent)"""

_usage = {"date": None, "count": 0}
_lock = threading.Lock()


@spaces.GPU
def _gpu_placeholder():
    """ZeroGPU Spaces need at least one GPU function. The agent itself runs on CPU."""


def _allow_question() -> bool:
    """Count questions per Copenhagen day. Resets at midnight."""
    today = datetime.now(LOCAL_TZ).date()
    with _lock:
        if _usage["date"] != today:
            _usage.update(date=today, count=0)
        if _usage["count"] >= DAILY_LIMIT:
            return False
        _usage["count"] += 1
        return True


def _describe_step(step) -> str:
    args = step.arguments
    if step.tool == "search_tables":
        text = f"Searched for *{args.get('query', '')}*"
    elif step.tool == "describe_table":
        text = f"Inspected table **{args.get('table_id', '')}**"
    elif step.tool == "find_values":
        matches = ", ".join(step.result.get("matches", [])[:3]) or "no match"
        text = f"Looked up *{args.get('text', '')}* in {args.get('variable_id', '')}: {matches}"
    elif step.tool == "get_data":
        selections = json.dumps(args.get("selections", {}), ensure_ascii=False)
        text = f"Fetched data from **{args.get('table_id', '')}** with `{selections}`"
    else:
        text = f"Called {step.tool}"
    if "error" in step.result:
        text += f" (error: {step.result['error']})"
    return text


def _format_trace(result: AgentResult) -> str:
    lines = [
        f"{i}. {_describe_step(step)}" for i, step in enumerate(result.steps, start=1)
    ]
    summary = f"{len(result.steps)} steps, {result.seconds:.0f} seconds"
    return f"**How I found this** ({summary}):\n\n" + "\n".join(lines)


def answer(question: str, history: list) -> str:
    question = (question or "").strip()
    if not question:
        return "Please ask a question."
    if len(question) > MAX_QUESTION_CHARS:
        return f"Please keep questions under {MAX_QUESTION_CHARS} characters."
    if not _allow_question():
        return (
            "The demo has reached its daily question limit. Please try again tomorrow."
        )
    try:
        result = run_agent(question)
    except LLMError:
        return "The free AI models are busy right now. Please try again in a minute."
    return f"{result.answer}\n\n---\n\n{_format_trace(result)}"


demo = gr.ChatInterface(
    fn=answer,
    title="Ask Statistics Denmark",
    description=DESCRIPTION,
    examples=EXAMPLES,
)

# One question at a time keeps the app within the free models' per-minute limits
demo.queue(default_concurrency_limit=1, max_size=10)

if __name__ == "__main__":
    demo.launch()
