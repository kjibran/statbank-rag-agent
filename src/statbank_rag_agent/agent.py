import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from statbank_rag_agent import llm
from statbank_rag_agent.tools import TOOL_SCHEMAS, run_tool

MAX_STEPS = 8
MAX_TOOL_RESULT_CHARS = 3000  # keeps long tool results from filling the model's context
LOCAL_TZ = ZoneInfo(
    "Europe/Copenhagen"
)  # the users' and Statistics Denmark's time zone

SYSTEM_PROMPT = """You answer questions about Denmark using official statistics from Statistics Denmark (Statbank).
Today's date is {today}.

Work step by step with the tools:
1. search_tables with a short topic, for example "population" or "unemployment rate". Leave out place names and specific values.
2. describe_table for the most promising table. Check that its variables and period fit the question. If they do not, describe another candidate or search again with other words. If a table does not fit after one or two attempts, go back to the search results and try the next candidate.
3. find_values to look up codes for places, categories or ages. Municipalities and regions are usually in a variable called region (OMRÅDE).
4. get_data with as few codes as possible.

Rules:
- Only state numbers that came from get_data. Never estimate or invent numbers.
- Variables you leave out are summed over. Never add a total (such as "All Denmark" or "Total") to its own parts.
- Check that the data covers what was asked. A table broken down by islands, parishes or other groups may not cover all of Denmark. For national figures, use a table with a total such as "All Denmark", or one without a geographic breakdown.
- If the question does not name a period, use the latest available period and say which one it is.
- Before saying that no table fits, search at least twice with different words. For a specific date, also try "monthly" or "quarterly".
- If no table fits after that, say that you could not find a suitable table. Never claim that Statistics Denmark does not publish something: your search may simply have missed it.
- Keep the answer short: the number with its unit and period, and one sentence of context if useful.
- End with the source line from get_data, for example "Source: Statistics Denmark, StatBank.dk/folk1a".
  If you combined or calculated values yourself, write "Source: Own calculations based on data from Statistics Denmark, StatBank.dk/<table>".
"""

FINAL_ANSWER_PROMPT = """You reached the step limit while answering this question:
{question}

These are the tool calls you made and what they returned:
{transcript}

Answer now using only these results. If they do not contain the answer, say that you could not find it.
End with the source line, as described in your instructions."""

REPEATED_CALL = {
    "error": "You already made this exact call. Do not repeat it. Try a different table from the "
    "search results, or search with different words."
}


@dataclass
class Step:
    tool: str
    arguments: dict
    result: dict
    provider: str


@dataclass
class AgentResult:
    question: str
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    hit_step_limit: bool = False
    restarts: int = 0  # how often the question was restarted with another provider
    repeated_calls: int = (
        0  # identical tool calls that were blocked instead of run again
    )
    tokens: int = 0  # total tokens used across all model calls
    seconds: float = 0.0

    @property
    def tables_used(self) -> list[str]:
        """Tables the agent actually fetched data from."""
        return sorted(
            {
                s.arguments.get("table_id", "").upper()
                for s in self.steps
                if s.tool == "get_data" and "error" not in s.result
            }
        )


def _parse_arguments(raw: str) -> dict:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _call_key(tool: str, arguments: dict) -> str:
    """Identical calls get identical keys, whatever the order of their arguments."""
    return f"{tool}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"


def _today() -> str:
    return datetime.now(LOCAL_TZ).date().isoformat()


def _system_message() -> dict:
    return {"role": "system", "content": SYSTEM_PROMPT.format(today=_today())}


def _transcript(steps: list[Step]) -> str:
    """The tool calls as plain text, for a final answer without any tool context."""
    lines = []
    for i, step in enumerate(steps, start=1):
        arguments = json.dumps(step.arguments, ensure_ascii=False)
        result = json.dumps(step.result, ensure_ascii=False)[:MAX_TOOL_RESULT_CHARS]
        lines.append(f"{i}. {step.tool}({arguments}) returned: {result}")
    return "\n".join(lines)


def _final_answer(provider: llm.Provider, question: str, result: AgentResult) -> None:
    """Force a text answer at the step limit.

    The conversation is rebuilt as plain text without tool calls. Some models keep trying
    to call tools when they see earlier tool calls, even when no tools are offered.
    """
    messages = [
        _system_message(),
        {
            "role": "user",
            "content": FINAL_ANSWER_PROMPT.format(
                question=question, transcript=_transcript(result.steps)
            ),
        },
    ]
    message, name, tokens = llm.chat(messages, provider=provider)
    result.providers.append(name)
    result.tokens += tokens
    result.answer = (message.get("content") or "").strip()


def _run_with(provider: llm.Provider, question: str, max_steps: int) -> AgentResult:
    """The whole conversation with one provider. Tool-calling conversations cannot switch midway."""
    result = AgentResult(question=question)
    messages = [_system_message(), {"role": "user", "content": question}]
    seen_calls: set[str] = set()

    for _ in range(max_steps):
        message, name, tokens = llm.chat(messages, TOOL_SCHEMAS, provider=provider)
        result.providers.append(name)
        result.tokens += tokens
        messages.append(llm.clean_assistant_message(message))

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            result.answer = (message.get("content") or "").strip()
            return result

        for call in tool_calls:
            tool = call["function"]["name"]
            arguments = _parse_arguments(call["function"].get("arguments") or "{}")
            key = _call_key(tool, arguments)
            if key in seen_calls:
                # A repeated call would return the same result again: block it and say why
                output = REPEATED_CALL
                result.repeated_calls += 1
            else:
                output = run_tool(tool, json.dumps(arguments, ensure_ascii=False))
                seen_calls.add(key)
            result.steps.append(Step(tool, arguments, output, name))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(output, ensure_ascii=False)[
                        :MAX_TOOL_RESULT_CHARS
                    ],
                }
            )

    result.hit_step_limit = True
    _final_answer(provider, question, result)
    return result


def run_agent(question: str, max_steps: int = MAX_STEPS) -> AgentResult:
    """Answer a question. If a provider fails for good, restart the question with the next one."""
    start = time.perf_counter()
    errors = []
    for restarts, provider in enumerate(llm.providers()):
        try:
            result = _run_with(provider, question, max_steps)
        except llm.LLMError as exc:
            errors.append(str(exc))
            print(
                f"  {provider.name} failed, restarting the question with the next provider"
            )
            continue
        result.restarts = restarts
        result.seconds = time.perf_counter() - start
        return result
    raise llm.LLMError("Every provider failed for this question. " + " | ".join(errors))
