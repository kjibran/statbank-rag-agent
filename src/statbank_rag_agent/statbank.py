import time

import httpx

from statbank_rag_agent.config import settings

RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


def _get(path: str, params: dict, attempts: int = 4) -> httpx.Response:
    """GET with retries for timeouts, rate limits and server errors."""
    url = f"{settings.statbank_url}/{path}"
    for attempt in range(attempts):
        try:
            response = httpx.get(url, params=params, timeout=60)
            if response.status_code not in RETRY_STATUSES:
                response.raise_for_status()
                return response
            reason = f"HTTP {response.status_code}"
        except httpx.TimeoutException:
            reason = "timeout"
        wait = 5 * 2**attempt  # 5, 10, 20, 40 seconds
        print(f"  {reason} for {path}, retrying in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"Statbank request failed after {attempts} attempts: {url}")


def fetch_tables() -> list[dict]:
    """All tables in Statbank Denmark, with title, variables and time range."""
    return _get("tables", {"lang": settings.language, "format": "JSON"}).json()


def fetch_tableinfo(table_id: str) -> dict:
    """Full metadata for one table: its variables and all their values."""
    return _get(
        f"tableinfo/{table_id}", {"lang": settings.language, "format": "JSON"}
    ).json()
