import json

from statbank_rag_agent.llm import chat

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

message, provider = chat(
    [{"role": "user", "content": "What is the weather in Copenhagen right now?"}], tools
)
print("Answered by:", provider)
print("Text:", message.get("content"))
print("Tool calls:", json.dumps(message.get("tool_calls"), indent=1))
