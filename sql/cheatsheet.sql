-- ============================================================
-- pg_embedder Cheatsheet (Pure Rust)
-- ============================================================

-- ===================
-- SETUP
-- ===================
CREATE EXTENSION pg_embedder;
SELECT embed_init();
SELECT embed_ready(), embed_info(), embed_dim();

-- ===================
-- GENERATE EMBEDDINGS
-- ===================
SELECT embed_encode('hello world');           -- float[]
SELECT embed_text('hello world');             -- text format

-- ===================
-- SIMILARITY (NO PGVECTOR NEEDED)
-- ===================

-- Cosine similarity: [-1, 1], higher = more similar
SELECT cosine_similarity(
    embed_encode('cat'),
    embed_encode('kitten')
);

-- Cosine distance: [0, 2], lower = more similar
SELECT cosine_distance(
    embed_encode('cat'),
    embed_encode('dog')
);

-- L2 distance: [0, ∞), lower = more similar
SELECT l2_distance(
    embed_encode('hello'),
    embed_encode('world')
);

-- Dot product (for normalized vectors)
SELECT dot_product(
    embed_encode('test'),
    embed_encode('test')
);

-- Direct text comparison
SELECT text_similarity('cat', 'kitten');

-- Which model is compiled into this build?
SELECT embed_model();

-- ===================
-- TABLES
-- ===================
CREATE TABLE docs (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding FLOAT4[]
);

-- Auto-embed trigger
CREATE OR REPLACE FUNCTION auto_embed()
RETURNS TRIGGER AS $$
BEGIN
    NEW.embedding := embed_encode(NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER docs_embed
    BEFORE INSERT OR UPDATE OF content ON docs
    FOR EACH ROW EXECUTE FUNCTION auto_embed();

-- ===================
-- SEARCH
-- ===================
-- Basic search
WITH query AS (SELECT embed_encode('search term') AS emb)
SELECT id, content, cosine_similarity(embedding, query.emb) AS score
FROM docs, query
ORDER BY score DESC
LIMIT 5;

-- With threshold
WITH query AS (SELECT embed_encode('search term') AS emb)
SELECT id, content, cosine_similarity(embedding, query.emb) AS score
FROM docs, query
WHERE cosine_similarity(embedding, query.emb) > 0.7
ORDER BY score DESC;

-- ===================
-- WITH PGVECTOR (OPTIONAL)
-- ===================
-- If pgvector is installed, you can use indexes:

-- CREATE EXTENSION vector;
-- ALTER TABLE docs ADD COLUMN vec vector(384);
-- UPDATE docs SET vec = embedding::vector(384);
-- CREATE INDEX ON docs USING hnsw (vec vector_cosine_ops);
-- 
-- -- Then use pgvector operators:
-- SELECT * FROM docs ORDER BY vec <=> embed_encode('query')::vector(384) LIMIT 5;