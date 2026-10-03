import time
from dataclasses import dataclass, field

import httpx

from statbank_rag_agent.config import settings

RATE_LIMIT_ATTEMPTS = 3
DEFAULT_RETRY_SECONDS = 20.0
MAX_RETRY_SECONDS = 60.0


class LLMError(Exception):
    """The provider (or every provider) failed for this request."""


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str
    extra: dict = field(default_factory=dict)  # provider-specific request options


def providers() -> list[Provider]:
    """Tool-capable providers in fallback order. Providers without a key or model are skipped."""
    candidates = [
        Provider(
            "groq",
            "https://api.groq.com/openai/v1",
            settings.groq_api_key,
            settings.groq_model,
            # gpt-oss models reason before answering, and reasoning tokens count against
            # the free per-minute budget. Low effort is enough for choosing tools.
            extra={"reasoning_effort": "low"},
        ),
        Provider(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            settings.gemini_api_key,
            settings.gemini_model,
        ),
    ]
    return [p for p in candidates if p.api_key and p.model]


def _retry_after(response: httpx.Response) -> float:
    """How long the provider asks us to wait, capped at a minute."""
    header = response.headers.get("retry-after")
    try:
        return (
            min(float(header), MAX_RETRY_SECONDS) if header else DEFAULT_RETRY_SECONDS
        )
    except ValueError:
        return DEFAULT_RETRY_SECONDS


def _call(
    provider: Provider, messages: list[dict], tools: list[dict] | None
) -> tuple[dict, int]:
    """One request to one provider. Returns the message and the tokens used.

    Rate limits and network errors are retried after a pause.
    """
    payload: dict = {
        "model": provider.model,
        "messages": messages,
        "temperature": 0,
        **provider.extra,
    }
    if tools:
        payload["tools"] = tools
    problem = ""
    for _ in range(RATE_LIMIT_ATTEMPTS):
        try:
            response = httpx.post(
                f"{provider.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {provider.api_key}"},
                timeout=60,
            )
        except httpx.HTTPError as exc:
            problem = f"{provider.name}: {type(exc).__name__}"
            time.sleep(5)
            continue
        if response.status_code == 429:
            wait = _retry_after(response)
            problem = f"{provider.name}: rate limited"
            print(f"  {provider.name} rate limit reached, waiting {wait:.0f}s")
            time.sleep(wait)
            continue
        if response.status_code != 200:
            raise LLMError(
                f"{provider.name}: HTTP {response.status_code}: {response.text[:200]}"
            )
        data = response.json()
        tokens = (data.get("usage") or {}).get("total_tokens", 0)
        return data["choices"][0]["message"], tokens
    raise LLMError(f"{problem} (gave up after {RATE_LIMIT_ATTEMPTS} attempts)")


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    provider: Provider | None = None,
) -> tuple[dict, str, int]:
    """Send a chat request. With `provider`, only that provider is used. Otherwise fall back in order.

    Returns the model's message, the provider that answered, and the tokens used.
    """
    candidates = [provider] if provider else providers()
    if not candidates:
        raise LLMError("No provider configured.")
    errors = []
    for candidate in candidates:
        try:
            message, tokens = _call(candidate, messages, tools)
            return message, candidate.name, tokens
        except LLMError as exc:
            errors.append(str(exc))
    raise LLMError("All providers failed. " + " | ".join(errors))


def clean_assistant_message(message: dict) -> dict:
    """Keep what a provider needs to continue the same conversation.

    Tool calls are kept exactly as returned: Gemini attaches a signature to each call and
    rejects later requests without it. Other extras, such as Groq's 'reasoning' text, are dropped.
    """
    cleaned = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        cleaned["tool_calls"] = message["tool_calls"]
    if message.get("extra_content"):
        cleaned["extra_content"] = message["extra_content"]
    return cleaned
