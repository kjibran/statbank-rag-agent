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
2. describe_table for the most promising table. Check that its variables and period fit the question. If they do not, describe another candidate or search again with other words.
3. find_values to look up codes for places, categories or ages. Municipalities and regions are usually in a variable called region (OMRÅDE).
4. get_data with as few codes as possible.

Rules:
- Only state numbers that came from get_data. Never estimate or invent numbers.
- Variables you leave out are summed over. Never add a total (such as "All Denmark" or "Total") to its own parts.
- Check that the data covers what was asked. A table broken down by islands, parishes or other groups may not cover all of Denmark. For national figures, use a table with a total such as "All Denmark", or one without a geographic breakdown.
- If the question does not name a period, use the latest available period and say which one it is.
- If no table fits the question, say so plainly instead of guessing.
- Keep the answer short: the number with its unit and period, and one sentence of context if useful.
- End with the source line from get_data, for example "Source: Statistics Denmark, StatBank.dk/folk1a".
  If you combined or calculated values yourself, write "Source: Own calculations based on data from Statistics Denmark, StatBank.dk/<table>".
"""


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


def _today() -> str:
    return datetime.now(LOCAL_TZ).date().isoformat()


def _run_with(provider: llm.Provider, question: str, max_steps: int) -> AgentResult:
    """The whole conversation with one provider. Tool-calling conversations cannot switch midway."""
    result = AgentResult(question=question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=_today())},
        {"role": "user", "content": question},
    ]

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
            raw_arguments = call["function"].get("arguments") or "{}"
            output = run_tool(tool, raw_arguments)
            result.steps.append(
                Step(tool, _parse_arguments(raw_arguments), output, name)
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(output, ensure_ascii=False)[
                        :MAX_TOOL_RESULT_CHARS
                    ],
                }
            )

    # Step limit reached: one last call without tools forces a text answer
    result.hit_step_limit = True
    messages.append(
        {
            "role": "user",
            "content": "You have reached the step limit. Answer now using only data you already "
            "fetched, or say that you could not find it.",
        }
    )
    message, name, tokens = llm.chat(messages, provider=provider)
    result.providers.append(name)
    result.tokens += tokens
    result.answer = (message.get("content") or "").strip()
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
