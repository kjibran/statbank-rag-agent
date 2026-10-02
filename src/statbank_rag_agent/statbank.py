import httpx

from statbank_rag_agent.config import settings


def fetch_tables() -> list[dict]:
    """All tables in Statbank Denmark, with title, variables and time range."""
    response = httpx.get(
        f"{settings.statbank_url}/tables",
        params={"lang": settings.language, "format": "JSON"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()
