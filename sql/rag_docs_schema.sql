create table if not exists documents (
    id text primary key, content text not null,
    metadata jsonb, created_at timestamptz default now());
create index if not exists documents_metadata_idx on documents using gin(metadata);