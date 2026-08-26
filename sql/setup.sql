-- ============================================================
-- pg_embedder Setup (Pure Rust - No Dependencies)
-- ============================================================

\echo '=== Setting up pg_embedder ==='

-- Enable extension
CREATE EXTENSION IF NOT EXISTS pg_embedder;

-- Initialize the embedding model
SELECT embed_init();

-- ============================================================
-- Check Status
-- ============================================================

CREATE OR REPLACE FUNCTION embed_status()
RETURNS TABLE(
    component TEXT,
    status TEXT,
    details TEXT
)
LANGUAGE SQL
AS $$
    SELECT 'pg_embedder'::TEXT, 
           CASE WHEN embed_ready() THEN 'ready' ELSE 'not initialized' END,
           embed_info()
    UNION ALL
    SELECT 'model'::TEXT,
           'embedded'::TEXT,
           embed_model()
    UNION ALL
    SELECT 'dimension'::TEXT,
           'configured'::TEXT,
           embed_dim()::TEXT || ' dimensions'
    UNION ALL
    SELECT 'pgvector'::TEXT,
           CASE WHEN EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') 
                THEN 'installed' ELSE 'not installed (optional)' END,
           'use ::vector(384) cast if available'
$$;

\echo ''
SELECT * FROM embed_status();

-- ============================================================
-- Usage Examples
-- ============================================================

\echo ''
\echo '=== Available Functions ==='
\echo ''
\echo 'Embedding generation:'
\echo '  embed_encode(text) -> float[]'
\echo '  embed_text(text) -> text (for casting)'
\echo ''
\echo 'Similarity (standalone, no pgvector needed):'
\echo '  cosine_similarity(float[], float[]) -> float  (higher = similar)'
\echo '  cosine_distance(float[], float[]) -> float    (lower = similar)'
\echo '  l2_distance(float[], float[]) -> float        (lower = similar)'
\echo '  dot_product(float[], float[]) -> float'
\echo '  text_similarity(text, text) -> float'
\echo ''
\echo 'Utilities:'
\echo '  embed_init() -> text'
\echo '  embed_ready() -> boolean'
\echo '  embed_info() -> text'
\echo '  embed_model() -> text'
\echo '  embed_dim() -> int'
\echo ''
\echo '=== Quick Test ==='
-- Near-synonyms should score well above an unrelated pair.
SELECT text_similarity('cat', 'kitten')  AS related,
       text_similarity('cat', 'bicycle') AS unrelated;
