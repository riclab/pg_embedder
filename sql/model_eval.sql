-- ============================================================
-- pg_embedder retrieval eval
--
-- Runs the same tiny retrieval benchmark against whatever model the loaded
-- build embeds, so two builds can be compared on identical inputs. Run it once
-- per build and diff the summary:
--
--   cargo pgrx install --release                                              # model-en
--   psql -d your_database -f sql/model_eval.sql > /tmp/eval-en.txt
--   cargo pgrx install --release --no-default-features --features pg17,model-en-arctic
--   psql -d your_database -f sql/model_eval.sql > /tmp/eval-arctic.txt
--   diff /tmp/eval-en.txt /tmp/eval-arctic.txt
--
-- Reconnect between runs: the model is loaded once per backend process, so an
-- open session keeps serving the previous build's weights.
--
-- Everything is TEMP; the script leaves no tables behind. Documents are encoded
-- with embed_encode() and queries with embed_query(), which is the distinction
-- asymmetric models are trained on.
-- ============================================================

\echo '=== model under test ==='
SELECT embed_model() AS model, embed_info() AS info;

CREATE TEMP TABLE eval_docs (
    id      INT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding FLOAT4[]
);

INSERT INTO eval_docs (id, content) VALUES
    (1,  'PostgreSQL stores rows in heap files and indexes them with B-trees.'),
    (2,  'Vector similarity search compares dense embeddings with cosine distance.'),
    (3,  'HNSW builds a navigable small world graph for approximate nearest neighbours.'),
    (4,  'A database transaction is atomic, consistent, isolated and durable.'),
    (5,  'Sourdough bread needs a live starter and a long cold fermentation.'),
    (6,  'Neural embeddings map text into a continuous space where meaning is geometry.'),
    (7,  'Connection pooling reduces the cost of opening new database backends.'),
    (8,  'The tokenizer splits text into subword units before the model sees it.'),
    (9,  'Olive trees prefer poor soil, full sun and very little water.'),
    (10, 'Write-ahead logging lets a database recover cleanly after a crash.'),
    (11, 'Chunking long documents keeps each passage inside the model context window.'),
    (12, 'Quantisation trades a little accuracy for a much smaller model footprint.');

UPDATE eval_docs SET embedding = embed_encode(content);

CREATE TEMP TABLE eval_queries (
    q         TEXT PRIMARY KEY,
    expect    INT NOT NULL,
    embedding FLOAT4[]
);

INSERT INTO eval_queries (q, expect) VALUES
    ('how does approximate nearest neighbour search work',   3),
    ('what is cosine distance used for',                     2),
    ('how do I recover a database after a crash',           10),
    ('why split a long document into pieces',               11),
    ('how is text turned into tokens',                       8),
    ('making the model smaller at some cost in accuracy',   12),
    ('reusing database connections instead of reopening',    7),
    ('how to bake bread with a natural starter',             5);

UPDATE eval_queries SET embedding = embed_query(q);

-- Rank every document per query, then look up where the expected one landed.
CREATE TEMP VIEW eval_ranked AS
SELECT
    q.q,
    q.expect,
    d.id,
    cosine_similarity(d.embedding, q.embedding) AS score,
    ROW_NUMBER() OVER (
        PARTITION BY q.q
        ORDER BY cosine_similarity(d.embedding, q.embedding) DESC
    ) AS rank
FROM eval_queries q
CROSS JOIN eval_docs d;

\echo ''
\echo '=== per query (rank 1 = expected document retrieved first) ==='
SELECT
    r.q                                    AS query,
    r.expect                               AS expected_doc,
    r.rank                                 AS rank_of_expected,
    ROUND(r.score::numeric, 4)             AS score,
    ROUND((r.score - (
        SELECT MAX(o.score) FROM eval_ranked o
        WHERE o.q = r.q AND o.id <> r.expect
    ))::numeric, 4)                        AS margin_over_best_wrong
FROM eval_ranked r
WHERE r.id = r.expect
ORDER BY r.rank, r.q;

\echo ''
\echo '=== summary (higher is better; compare across builds) ==='
SELECT
    embed_model()                                              AS model,
    COUNT(*)                                                   AS queries,
    SUM((rank = 1)::int)                                       AS top1_hits,
    ROUND(AVG((rank = 1)::int)::numeric, 4)                    AS top1_accuracy,
    ROUND(AVG(1.0 / rank)::numeric, 4)                         AS mrr,
    ROUND(AVG(rank)::numeric, 3)                               AS mean_rank
FROM eval_ranked
WHERE id = expect;
