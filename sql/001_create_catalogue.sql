-- Catalogue of Statistics Denmark tables, searchable by keywords and by meaning.
create extension if not exists vector;

create table statbank_tables (
    table_id         text        primary key,  -- e.g. 'FOLK1A'
    title            text        not null,
    unit             text,
    first_period     text,                     -- e.g. '2008Q1'
    latest_period    text,
    variables        text[]      not null,     -- English variable labels
    search_text      text        not null,     -- title, variables, period and unit in one text
    embedding        vector(384),              -- null until computed
    embedding_model  text,                     -- which model produced the embedding
    source_updated   timestamptz,              -- when Statistics Denmark last updated the table
    ingested_at      timestamptz not null default now()
);

-- Meaning-based search: approximate nearest neighbours by cosine distance
create index statbank_tables_embedding_idx
    on statbank_tables using hnsw (embedding vector_cosine_ops);

-- Keyword search: English full-text index on the same text
create index statbank_tables_search_idx
    on statbank_tables using gin (to_tsvector('english', search_text));

alter table statbank_tables enable row level security;