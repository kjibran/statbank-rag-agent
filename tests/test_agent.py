import pytest

from statbank_rag_agent import agent, llm

PROVIDER_A = llm.Provider("a", "http://fake", "key", "model")
PROVIDER_B = llm.Provider("b", "http://fake", "key", "model")


def tool_call(call_id, name, arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def answer(text):
    return {"role": "assistant", "content": text}


@pytest.fixture
def one_provider(monkeypatch):
    monkeypatch.setattr(agent.llm, "providers", lambda: [PROVIDER_A])
    monkeypatch.setattr(agent, "run_tool", lambda name, arguments: {"ok": True})


def test_agent_runs_tools_then_answers(monkeypatch, one_provider):
    replies = iter(
        [
            tool_call("c1", "describe_table", '{"table_id": "FOLK1A"}'),
            answer("42 people."),
        ]
    )
    monkeypatch.setattr(
        agent.llm,
        "chat",
        lambda messages, tools=None, provider=None: (next(replies), provider.name, 100),
    )

    result = agent.run_agent("How many?")
    assert result.answer == "42 people."
    assert [s.tool for s in result.steps] == ["describe_table"]
    assert result.steps[0].arguments == {"table_id": "FOLK1A"}
    assert result.providers == ["a", "a"]
    assert result.tokens == 200  # two model calls of 100 tokens each
    assert result.restarts == 0


def test_step_limit_forces_a_final_answer_without_tool_context(
    monkeypatch, one_provider
):
    final_messages = []

    def fake_chat(messages, tools=None, provider=None):
        if tools is None:  # the final call
            final_messages.extend(messages)
            return answer("I could not find it."), provider.name, 10
        return tool_call("c", "search_tables", '{"query": "x"}'), provider.name, 10

    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    result = agent.run_agent("Impossible question", max_steps=3)
    assert result.hit_step_limit
    assert len(result.steps) == 3
    assert result.answer == "I could not find it."
    # The final call sees the tool results as plain text, never as tool calls
    assert all(m["role"] in {"system", "user"} for m in final_messages)
    assert "search_tables" in final_messages[-1]["content"]


def test_question_restarts_with_next_provider_instead_of_switching_midway(monkeypatch):
    monkeypatch.setattr(agent.llm, "providers", lambda: [PROVIDER_A, PROVIDER_B])
    monkeypatch.setattr(agent, "run_tool", lambda name, arguments: {"ok": True})
    calls = {"a": 0}

    def fake_chat(messages, tools=None, provider=None):
        if provider.name == "a":
            calls["a"] += 1
            if calls["a"] == 2:  # provider a fails in the middle of the conversation
                raise llm.LLMError("a: rate limited")
            return tool_call("c", "search_tables", '{"query": "x"}'), "a", 10
        if len(messages) == 2:  # b starts from scratch: only system prompt and question
            return tool_call("c", "search_tables", '{"query": "x"}'), "b", 10
        return answer("Done."), "b", 10

    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    result = agent.run_agent("q")
    assert result.answer == "Done."
    assert result.providers == ["b", "b"]  # nothing from a carries over
    assert result.restarts == 1


def test_tables_used_only_counts_successful_data_fetches():
    result = agent.AgentResult(question="q")
    result.steps = [
        agent.Step("get_data", {"table_id": "folk1a"}, {"rows": []}, "fake"),
        agent.Step("get_data", {"table_id": "bef5"}, {"error": "bad code"}, "fake"),
        agent.Step("describe_table", {"table_id": "BEFOLK2"}, {}, "fake"),
    ]
    assert result.tables_used == ["FOLK1A"]


def test_identical_tool_calls_are_blocked_not_repeated(monkeypatch, one_provider):
    executed = []
    monkeypatch.setattr(
        agent,
        "run_tool",
        lambda name, arguments: executed.append(arguments) or {"matches": []},
    )
    same_call = tool_call(
        "c",
        "find_values",
        '{"table_id": "BEF4", "variable_id": "OER", "text": "Denmark"}',
    )
    # Same arguments in a different order: still the same call
    reordered = tool_call(
        "d",
        "find_values",
        '{"text": "Denmark", "table_id": "BEF4", "variable_id": "OER"}',
    )
    replies = iter([same_call, reordered, answer("I could not find it.")])
    monkeypatch.setattr(
        agent.llm,
        "chat",
        lambda messages, tools=None, provider=None: (next(replies), provider.name, 10),
    )

    result = agent.run_agent("q")
    assert len(executed) == 1  # the tool ran once
    assert result.repeated_calls == 1
    assert "already made this exact call" in result.steps[1].result["error"]
