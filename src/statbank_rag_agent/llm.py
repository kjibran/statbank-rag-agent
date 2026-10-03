from dataclasses import dataclass

import httpx

from statbank_rag_agent.config import settings


class LLMError(Exception):
    """Every provider failed for this request."""


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str


def providers() -> list[Provider]:
    """Tool-capable providers in fallback order. Providers without a key or model are skipped."""
    candidates = [
        Provider(
            "groq",
            "https://api.groq.com/openai/v1",
            settings.groq_api_key,
            settings.groq_model,
        ),
        Provider(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            settings.gemini_api_key,
            settings.gemini_model,
        ),
    ]
    return [p for p in candidates if p.api_key and p.model]


def chat(messages: list[dict], tools: list[dict] | None = None) -> tuple[dict, str]:
    """Send a chat request, trying providers in order.

    Returns the model's message (text, or a request to call tools) and the provider that answered.
    """
    errors = []
    for provider in providers():
        payload: dict = {
            "model": provider.model,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            payload["tools"] = tools
        try:
            response = httpx.post(
                f"{provider.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {provider.api_key}"},
                timeout=60,
            )
        except httpx.HTTPError as exc:
            errors.append(f"{provider.name}: {type(exc).__name__}")
            continue
        if response.status_code != 200:
            errors.append(
                f"{provider.name}: HTTP {response.status_code}: {response.text[:200]}"
            )
            continue
        return response.json()["choices"][0]["message"], provider.name
    raise LLMError(
        "All providers failed. " + " | ".join(errors)
        if errors
        else "No provider configured."
    )


def clean_assistant_message(message: dict) -> dict:
    """Keep only the standard fields when a model reply goes back into the conversation.

    Providers add their own extras (Groq returns a 'reasoning' field, for example), and
    sending those back can make another provider reject the request after a fallback.
    """
    cleaned = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        cleaned["tool_calls"] = message["tool_calls"]
    return cleaned
