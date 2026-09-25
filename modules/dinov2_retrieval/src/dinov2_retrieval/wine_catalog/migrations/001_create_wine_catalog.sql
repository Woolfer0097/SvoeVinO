CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS wines (
    id BIGSERIAL PRIMARY KEY,
    name TEXT,
    category TEXT,
    color TEXT,
    region TEXT,
    grape_variety TEXT,
    description TEXT,
    winery TEXT,
    slug TEXT,
    dataset_photo TEXT,
    web_photo TEXT,
    web_photo_url TEXT,
    dataset_photo_embedding VECTOR,
    description_text_embedding VECTOR,
    web_photo_embedding VECTOR,
    source_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    import_source TEXT NOT NULL,
    source_row_number BIGINT NOT NULL,
    CONSTRAINT wines_import_source_row_key
        UNIQUE (import_source, source_row_number)
);

CREATE INDEX IF NOT EXISTS wines_slug_idx ON wines (slug);

CREATE TABLE IF NOT EXISTS wine_scrape_jobs (
    slug TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'in_progress', 'succeeded', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    processed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
