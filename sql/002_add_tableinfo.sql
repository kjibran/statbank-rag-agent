-- Full table metadata (variables and their values) from the Statbank tableinfo endpoint.
alter table statbank_tables
    add column tableinfo jsonb,
    add column tableinfo_fetched_at timestamptz;