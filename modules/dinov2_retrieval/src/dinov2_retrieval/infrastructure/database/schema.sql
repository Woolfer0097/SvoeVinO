-- Reference photo embeddings for visual wine search.
-- database/init.sql and src/dinov2_retrieval/infrastructure/database/schema.sql
-- must stay identical (a unit test checks this).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS reference_images (
    id BIGSERIAL PRIMARY KEY,
    wine_id TEXT NOT NULL,
    slug TEXT NOT NULL,
    image_uri TEXT NOT NULL UNIQUE,
    model_name TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
