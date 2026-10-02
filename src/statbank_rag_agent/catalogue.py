from datetime import datetime
from zoneinfo import ZoneInfo

from psycopg.types.json import Jsonb

from statbank_rag_agent.db import connect

STATBANK_TZ = ZoneInfo("Europe/Copenhagen")
MAX_VALUES_PER_VARIABLE = 12


def build_search_text(table: dict) -> str:
    """Basic search text from the table list: title, variables, time range and unit.

    Used only for newly added tables, until their full metadata has been fetched.
    """
    variables = ", ".join(table.get("variables") or [])
    return (
        f"{table['text']}. "
        f"Variables: {variables}. "
        f"Covers {table.get('firstPeriod')} to {table.get('latestPeriod')}. "
        f"Unit: {table.get('unit')}."
    )


def build_rich_search_text(
    info: dict, first_period: str | None, latest_period: str | None
) -> str:
    """Search text from full metadata, including value labels for each variable.

    Value labels carry words people search for (such as "retail" or "dentist")
    and tell apart tables whose titles and variable names are identical.
    """
    parts = [
        f"{info['text']}.",
        f"Covers {first_period} to {latest_period}.",
        f"Unit: {info.get('unit')}.",
    ]
    for variable in info.get("variables", []):
        if variable.get("time"):
            continue  # time is already covered by the period
        labels = [value["text"] for value in variable.get("values", [])]
        sample = ", ".join(labels[:MAX_VALUES_PER_VARIABLE])
        more = len(labels) - MAX_VALUES_PER_VARIABLE
        suffix = f" and {more} more" if more > 0 else ""
        parts.append(f"{variable['text']}: {sample}{suffix}.")
    return " ".join(parts)


def _parse_updated(value: str | None) -> datetime | None:
    """Statistics Denmark gives local Copenhagen times without a time zone."""
    if not value:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=STATBANK_TZ)


def to_row(table: dict) -> dict:
    return {
        "table_id": table["id"],
        "title": table["text"],
        "unit": table.get("unit"),
        "first_period": table.get("firstPeriod"),
        "latest_period": table.get("latestPeriod"),
        "variables": table.get("variables") or [],
        "search_text": build_search_text(table),
        "source_updated": _parse_updated(table.get("updated")),
    }


# The basic search text is only written for new tables. Existing tables keep their
# rich search text, which is maintained by rebuild_search_texts().
UPSERT_TABLE = """
insert into statbank_tables (
    table_id, title, unit, first_period, latest_period, variables, search_text, source_updated
)
values (
    %(table_id)s, %(title)s, %(unit)s, %(first_period)s, %(latest_period)s,
    %(variables)s, %(search_text)s, %(source_updated)s
)
on conflict (table_id) do update set
    title = excluded.title,
    unit = excluded.unit,
    first_period = excluded.first_period,
    latest_period = excluded.latest_period,
    variables = excluded.variables,
    source_updated = excluded.source_updated,
    ingested_at = now()
"""


def upsert_tables(rows: list[dict]) -> int:
    """Insert new tables and update existing ones. Safe to run repeatedly."""
    if not rows:
        return 0
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(UPSERT_TABLE, rows)
    return len(rows)


def tables_needing_tableinfo() -> list[str]:
    """Tables never fetched, or updated by Statistics Denmark since the last fetch."""
    with connect() as conn:
        rows = conn.execute(
            "select table_id from statbank_tables "
            "where tableinfo is null or tableinfo_fetched_at < source_updated "
            "order by table_id"
        ).fetchall()
    return [r[0] for r in rows]


def save_tableinfo(table_id: str, info: dict) -> None:
    with connect() as conn:
        conn.execute(
            "update statbank_tables set tableinfo = %s, tableinfo_fetched_at = now() "
            "where table_id = %s",
            (Jsonb(info), table_id),
        )


def rebuild_search_texts() -> tuple[int, int]:
    """Rebuild search texts from stored metadata. Clears embeddings only where the text changed.

    Returns (tables checked, tables changed).
    """
    with connect() as conn:
        rows = conn.execute(
            "select table_id, tableinfo, first_period, latest_period, search_text "
            "from statbank_tables where tableinfo is not null"
        ).fetchall()

        changed = []
        for table_id, info, first_period, latest_period, old_text in rows:
            new_text = build_rich_search_text(info, first_period, latest_period)
            if new_text != old_text:
                changed.append({"table_id": table_id, "search_text": new_text})

        with conn.cursor() as cur:
            cur.executemany(
                "update statbank_tables "
                "set search_text = %(search_text)s, embedding = null, embedding_model = null "
                "where table_id = %(table_id)s",
                changed,
            )
    return len(rows), len(changed)
