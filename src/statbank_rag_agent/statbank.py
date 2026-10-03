import csv
import io
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


def fetch_data(table_id: str, selections: dict[str, list[str]]) -> list[dict]:
    """Numbers for the selected codes. Variables not selected are summed over by Statistics Denmark."""
    body = {
        "table": table_id,
        "format": "CSV",
        "lang": settings.language,
        "variables": [
            {"code": code, "values": values} for code, values in selections.items()
        ],
    }
    response = httpx.post(f"{settings.statbank_url}/data", json=body, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            f"Statbank returned HTTP {response.status_code}: {response.text[:300]}"
        )

    # The CSV starts with a byte-order mark and uses semicolons. 'utf-8-sig' removes the mark.
    text = response.content.decode("utf-8-sig")
    rows = []
    for row in csv.DictReader(io.StringIO(text), delimiter=";"):
        value = row.pop("INDHOLD", None)  # Danish for "content": the number itself
        rows.append({**row, "value": value})
    return rows
