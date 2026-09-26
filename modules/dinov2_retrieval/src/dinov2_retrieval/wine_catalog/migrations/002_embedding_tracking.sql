ALTER TABLE wines
    ADD COLUMN IF NOT EXISTS dataset_photo_embedding_model TEXT,
    ADD COLUMN IF NOT EXISTS description_text_embedding_model TEXT,
    ADD COLUMN IF NOT EXISTS web_photo_embedding_model TEXT;

CREATE TABLE IF NOT EXISTS wine_embedding_jobs (
    wine_id BIGINT NOT NULL REFERENCES wines(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('dataset_photo', 'web_photo', 'description_text')),
    status TEXT NOT NULL CHECK (status IN ('in_progress', 'succeeded', 'failed')),
    model_name TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (wine_id, kind)
);
