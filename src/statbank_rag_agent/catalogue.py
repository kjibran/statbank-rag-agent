from datetime import datetime
from zoneinfo import ZoneInfo

from psycopg.types.json import Jsonb

from statbank_rag_agent.db import connect

STATBANK_TZ = ZoneInfo("Europe/Copenhagen")


def build_search_text(table: dict) -> str:
    """One text per table combining title, variables, time range and unit.

    Many tables share the same title, so variables and time range are what
    tell them apart in search.
    """
    variables = ", ".join(table.get("variables") or [])
    return (
        f"{table['text']}. "
        f"Variables: {variables}. "
        f"Covers {table.get('firstPeriod')} to {table.get('latestPeriod')}. "
        f"Unit: {table.get('unit')}."
    )


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
    -- a changed search text makes the old embedding stale, so clear it
    embedding = case
        when statbank_tables.search_text = excluded.search_text then statbank_tables.embedding
    end,
    embedding_model = case
        when statbank_tables.search_text = excluded.search_text then statbank_tables.embedding_model
    end,
    search_text = excluded.search_text,
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
