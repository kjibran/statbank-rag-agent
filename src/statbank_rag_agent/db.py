import psycopg

from statbank_rag_agent.config import settings


def connect(read_only: bool = False) -> psycopg.Connection:
    """Open a connection to the Supabase Postgres database.

    With read_only=True, Postgres itself refuses any write on this connection.
    """
    conn = psycopg.connect(settings.database_url)
    conn.read_only = read_only
    return conn
