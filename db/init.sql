-- init.sql - Database schema for VisionMap
-- Runs once when the Postgres container is first created.

-- Enable the pgvector extension (adds vector column type + distance operators)
CREATE EXTENSION IF NOT EXISTS vector;

-- Main table: stores image metadata alongside their feature embeddings.
-- I used vector(512) because the ResNet18 avgpool layer I'm using
-- outputs exactly 512 dimensions.
CREATE TABLE IF NOT EXISTS spatial_features (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_name  TEXT NOT NULL,
    embedding   vector(512) NOT NULL,
    latitude    DOUBLE PRECISION,
    longitude   DOUBLE PRECISION,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- HNSW index for approximate nearest-neighbor search on embeddings.
--
-- Why I chose HNSW over IVFFlat:
--   - IVFFlat clusters vectors via k-means: it's fast to build, but recall
--     degrades as data grows and I'd need to rebuild it after large inserts.
--   - HNSW builds a multi-layer navigable graph: it has a slower initial build,
--     but gives me consistently high recall (>95%) and handles incremental inserts natively.
--
-- Tuning params I decided to use:
--   m = 16             -> max edges per node (higher = better recall, more memory)
--   ef_construction = 64 -> search width during index build
--   (If I need to tweak it at query time, I can use SET hnsw.ef_search = N)
CREATE INDEX IF NOT EXISTS idx_spatial_features_embedding
    ON spatial_features
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
