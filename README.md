# pg_embedder

`pg_embedder` is a PostgreSQL extension for generating neural text embeddings directly inside PostgreSQL. It is written in Rust with [`pgrx`](https://github.com/pgcentralfoundation/pgrx), runs inference with the Candle ML stack, and can be used either standalone with `FLOAT4[]` columns or together with [`pgvector`](https://github.com/pgvector/pgvector) for indexed vector search.

The extension embeds the model files into the compiled shared library, so the database server does not need to load model assets from the filesystem at runtime. The default build embeds the official English model [`mixedbread-ai/mxbai-embed-xsmall-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1); a Spanish variant is available behind a Cargo feature.

## Features

- Generate 384-dimensional text embeddings from SQL.
- Store embeddings as plain PostgreSQL `FLOAT4[]` arrays without extra extensions.
- Compute cosine similarity, cosine distance, L2 distance, and dot product in SQL.
- Compare two texts directly with `text_similarity(text, text)`.
- Optionally cast embeddings to `vector(384)` and use pgvector indexes/operators.
- Run CPU-only inference with model weights embedded into the extension binary.
- Choose the embedded model at build time: English (default) or Spanish.
- Support PostgreSQL 13 through 18 via `pgrx` feature flags.

## Project Status

This repository currently builds `pg_embedder` version `0.2.0`.

The extension exposes a compact SQL API and includes example scripts under `sql/`. The current implementation is CPU-only and embeds a single 384-dimensional model chosen at build time.

## Repository Layout

```text
.
|-- Cargo.toml              # Rust package and pgrx/PostgreSQL feature configuration
|-- pg_embedder.control     # PostgreSQL extension control file
|-- src/
|   |-- lib.rs              # SQL-facing pgrx functions
|   |-- bin/pgrx_embed.rs   # pgrx embed entry point
|   `-- engine/mod.rs       # Candle/tokenizers embedding engine
|-- sql/
|   |-- setup.sql           # Extension setup and status helper
|   |-- examples.sql        # Standalone FLOAT4[] semantic search example
|   |-- cheatsheet.sql      # Quick SQL snippets
|   `-- rag_example.sql     # pgvector-oriented RAG example
|-- tests/pg_regress/       # pgrx regression setup
`-- weights/
    |-- SOURCES.md          # Model provenance and checksums
    |-- en/                 # Official English model (default)
    `-- es/                 # Spanish variant (model-es feature)
```

## Requirements

- Rust toolchain with Cargo.
- PostgreSQL development headers and `pg_config` for the PostgreSQL version you are targeting.
- `cargo-pgrx` `0.16.1`.
- PostgreSQL 13, 14, 15, 16, 17, or 18.
- Optional: pgvector if you want HNSW/IVFFlat indexes and vector operators.

Install `cargo-pgrx`:

```bash
cargo install cargo-pgrx --version 0.16.1 --locked
```

Initialize `pgrx` for your local PostgreSQL installation if you have not already done so:

```bash
cargo pgrx init --pg17 /path/to/pg_config
```

Use the PostgreSQL version that matches your environment. This crate defaults to PostgreSQL 17, but feature flags are available for `pg13`, `pg14`, `pg15`, `pg16`, `pg17`, and `pg18`.

## Build

Build with the default PostgreSQL 17 feature:

```bash
cargo build --release
```

Build for a specific PostgreSQL version:

```bash
cargo build --release --no-default-features --features pg16
```

For local extension development, run PostgreSQL through `pgrx`:

```bash
cargo pgrx run pg17
```

Install the extension into a configured PostgreSQL installation:

```bash
cargo pgrx install --release
```

If you target a non-default PostgreSQL version, pass the matching feature:

```bash
cargo pgrx install --release --no-default-features --features pg16
```

## Model Selection

The embedded model is chosen with a Cargo feature. English is the default, and is also
what a `--no-default-features` build links, so only Spanish requires an explicit flag.

| Model | Cargo feature | Weights | Payload | Notes |
| --- | --- | --- | --- | --- |
| `mixedbread-ai/mxbai-embed-xsmall-v1` | `model-en` (default) | `weights/en/` | 48.9 MB | Official upstream release, unmodified |
| Spanish variant | `model-es` | `weights/es/` | 48.9 MB | Same architecture, BF16 re-save |
| `Snowflake/snowflake-arctic-embed-xs` | `model-en-arctic` | `weights/en-arctic/` | 45.1 MB | Retrieval-tuned English, **asymmetric**, F16 cast of upstream |

Build with a non-default model:

```bash
cargo pgrx install --release --no-default-features --features pg17,model-es
cargo pgrx install --release --no-default-features --features pg17,model-en-arctic
```

`--no-default-features` is required: without it `model-en` stays on and enabling a second
model is a compile error. Ask a running database which
model it has:

```sql
SELECT embed_model();
```

All three are 384-dimensional over the same vocabulary, but their embeddings are **not
interchangeable**: switching models invalidates every embedding already stored in your
tables, so re-embed after a switch. See `weights/SOURCES.md` for provenance, checksums,
and how to add another model.

### Symmetric and asymmetric models

Some models are trained to encode a search query differently from a document. `model-en`
and `model-es` are symmetric — both sides are encoded identically. `model-en-arctic` is
asymmetric: queries carry an instruction prefix that documents must not have, and
applying it to documents costs retrieval quality.

`embed_query()` handles this. It applies whatever the loaded model needs and is exactly
`embed_encode()` on symmetric models, so **search SQL written against `embed_query()`
stays correct across a model swap**:

```sql
-- index side
UPDATE docs SET embedding = embed_encode(content);

-- search side, correct for every model
SELECT id, content, cosine_similarity(embedding, embed_query('how does vector search work')) AS score
FROM docs
ORDER BY score DESC
LIMIT 5;
```

`SELECT embed_is_asymmetric();` reports which kind you have, and `embed_info()` shows the
pooling strategy alongside it.

### Comparing models on your own data

`sql/model_eval.sql` runs a small retrieval benchmark against whichever model the loaded
build embeds, reporting top-1 accuracy, MRR and mean rank. Run it once per build and diff:

```bash
cargo pgrx install --release
psql -d your_database -f sql/model_eval.sql > /tmp/eval-en.txt
cargo pgrx install --release --no-default-features --features pg17,model-en-arctic
psql -d your_database -f sql/model_eval.sql > /tmp/eval-arctic.txt
diff /tmp/eval-en.txt /tmp/eval-arctic.txt
```

Reconnect between runs: the model is loaded once per backend process, so an open session
keeps serving the previous build's weights.

## Database Setup

After installing or starting a `pgrx` development database, enable the extension:

```sql
CREATE EXTENSION pg_embedder;
```

Initialize the model for the current PostgreSQL backend/session:

```sql
SELECT embed_init();
```

Check status and metadata:

```sql
SELECT embed_ready();
SELECT embed_info();
SELECT embed_model();
SELECT embed_dim();
SELECT embed_version();
```

You can also run the setup helper script:

```bash
psql -d your_database -f sql/setup.sql
```

## Quick Start

Generate an embedding as a `FLOAT4[]` array:

```sql
SELECT embed_encode('hello world');
```

Generate an embedding as vector-literal text, useful for pgvector casting:

```sql
SELECT embed_text('hello world');
SELECT embed_text('hello world')::vector(384);
```

Compare two texts directly:

```sql
SELECT text_similarity('cat', 'kitten') AS similarity;
```

Compare embeddings manually:

```sql
SELECT cosine_similarity(
    embed_encode('wireless headphones'),
    embed_encode('bluetooth earbuds')
) AS similarity;
```

## SQL API

### Model Lifecycle

| Function | Returns | Description |
| --- | --- | --- |
| `embed_init()` | `text` | Loads the embedded model for the current backend/session if it is not already loaded. |
| `embed_ready()` | `boolean` | Returns whether the model has been loaded in the current backend/session. |
| `embed_info()` | `text` | Returns model name, dimension, load status, and mode information. |
| `embed_model()` | `text` | Returns the name of the embedding model compiled into this build. |
| `embed_is_asymmetric()` | `boolean` | Whether this build's model encodes queries differently from documents. |
| `embed_dim()` | `integer` | Returns the embedding dimension, currently `384`. |
| `embed_version()` | `text` | Returns the extension version string. |

### Embedding Generation

| Function | Returns | Description |
| --- | --- | --- |
| `embed_encode(text)` | `float4[]` | Encodes one input string into a normalized 384-dimensional embedding. This is the document side of a retrieval pair. |
| `embed_query(text)` | `float4[]` | Encodes a search query, applying the model's query prefix if it has one. Identical to `embed_encode()` on symmetric models. |
| `embed_text(text)` | `text` | Encodes text and returns a pgvector-compatible vector literal string. |
| `embed_query_text(text)` | `text` | Same as `embed_query()`, returned as a pgvector literal. |
| `embed_encode_batch(text[])` | `float4[][]` | Encodes multiple strings and returns one embedding array per input. |

### Similarity And Distance

| Function | Returns | Description |
| --- | --- | --- |
| `cosine_similarity(float4[], float4[])` | `float4` | Cosine similarity in `[-1, 1]`; higher means more similar. |
| `cosine_distance(float4[], float4[])` | `float4` | `1 - cosine_similarity`; lower means more similar. |
| `l2_distance(float4[], float4[])` | `float4` | Euclidean distance; lower means more similar. |
| `dot_product(float4[], float4[])` | `float4` | Dot product. For normalized vectors, higher usually means more similar. |
| `text_similarity(text, text)` | `float4` | Encodes both inputs and returns cosine similarity. |

## Standalone Semantic Search

You can use `pg_embedder` without pgvector by storing embeddings in a `FLOAT4[]` column.

```sql
CREATE EXTENSION IF NOT EXISTS pg_embedder;
SELECT embed_init();

CREATE TABLE docs (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding FLOAT4[]
);
```

Create a trigger that automatically embeds inserted or updated content:

```sql
CREATE OR REPLACE FUNCTION docs_auto_embed()
RETURNS TRIGGER AS $$
BEGIN
    NEW.embedding := embed_encode(NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER docs_embed_trigger
    BEFORE INSERT OR UPDATE OF content ON docs
    FOR EACH ROW
    EXECUTE FUNCTION docs_auto_embed();
```

Insert data:

```sql
INSERT INTO docs (content) VALUES
    ('PostgreSQL is a powerful open source relational database.'),
    ('Vector search compares dense numerical representations.'),
    ('Neural embeddings capture semantic meaning in text.');
```

Search by semantic similarity:

```sql
WITH query AS (
    SELECT embed_encode('how does vector search work') AS embedding
)
SELECT
    docs.id,
    docs.content,
    cosine_similarity(docs.embedding, query.embedding) AS score
FROM docs, query
ORDER BY score DESC
LIMIT 5;
```

Run the included standalone product-search example:

```bash
psql -d your_database -f sql/examples.sql
```

## pgvector Integration

pgvector is optional. Use it when you want indexed approximate nearest-neighbor search or pgvector distance operators.

Enable pgvector and `pg_embedder`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_embedder;
SELECT embed_init();
```

Create a table with a vector column:

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT NOT NULL,
    embedding vector(384)
);
```

Insert embeddings with `embed_text(... )::vector(384)`:

```sql
INSERT INTO documents (title, content, embedding)
VALUES (
    'Vector Search Basics',
    'Vector search enables semantic similarity queries over embeddings.',
    embed_text('Vector search enables semantic similarity queries over embeddings.')::vector(384)
);
```

Create an HNSW cosine index:

```sql
CREATE INDEX documents_embedding_hnsw_idx
    ON documents
    USING hnsw (embedding vector_cosine_ops);
```

Search with pgvector cosine distance:

```sql
SELECT
    id,
    title,
    content,
    1 - (embedding <=> embed_text('semantic search in databases')::vector(384)) AS similarity
FROM documents
ORDER BY embedding <=> embed_text('semantic search in databases')::vector(384)
LIMIT 5;
```

If your pgvector installation supports casting from `FLOAT4[]`, this also works:

```sql
SELECT *
FROM documents
ORDER BY embedding <=> embed_encode('semantic search in databases')::vector(384)
LIMIT 5;
```

`embed_text(... )::vector(384)` is the most portable form because it returns the vector literal format expected by pgvector.

## RAG Usage

The repository includes `sql/rag_example.sql` for Retrieval-Augmented Generation style document chunk retrieval. That script is designed around pgvector.

Before running it, define a small helper if your database does not already have one:

```sql
CREATE OR REPLACE FUNCTION to_embedding(input TEXT)
RETURNS vector(384)
LANGUAGE SQL
IMMUTABLE
AS $$
    SELECT embed_text(input)::vector(384)
$$;
```

Then run:

```bash
psql -d your_database -f sql/rag_example.sql
```

The RAG example creates a `documents` table, stores chunk embeddings, creates an HNSW index, and provides search functions that retrieve the most relevant chunks for a query.

## Model Behavior

- The default build embeds `mixedbread-ai/mxbai-embed-xsmall-v1` (English). `SELECT embed_model();` reports what a given build contains.
- Embeddings are 384-dimensional.
- Outputs are L2-normalized by the engine.
- Inputs are tokenized and truncated to a maximum sequence length of 512 tokens.
- Empty input passed to `embed_encode('')` returns a zero vector and emits a PostgreSQL warning.
- If model initialization or encoding fails, SQL functions emit warnings and return zero vectors where applicable.
- The model is held in a `OnceCell<Mutex<...>>` inside the PostgreSQL backend process. New database backend processes may need to initialize their own model instance.

## Performance Notes

- Inference is CPU-only in the current implementation.
- The engine sets `OMP_NUM_THREADS=1` and `RAYON_NUM_THREADS=1` when the model is initialized to avoid oversubscribing PostgreSQL backend processes.
- The model weights are embedded with `include_bytes!`, adding roughly 48 MB to the compiled extension but removing any runtime model-loading dependency. Combined with the release profile (`lto = "fat"`, `codegen-units = 1`), release builds are slow; use `cargo pgrx run` while iterating.
- For large tables, use pgvector with an HNSW or IVFFlat index instead of scanning `FLOAT4[]` arrays with SQL distance functions.
- For bulk ingestion, consider inserting source rows first and backfilling embeddings in batches rather than embedding everything in a trigger on a latency-sensitive write path.

## Testing

Run Rust tests:

```bash
cargo test
```

Run pgrx/PostgreSQL tests for the default PostgreSQL version:

```bash
cargo pgrx test pg17
```

Run tests for another supported PostgreSQL version:

```bash
cargo pgrx test pg16 --no-default-features --features pg16
```

The in-extension tests cover version reporting, embedding dimension, model initialization, text encoding, empty input behavior, and similarity functions.

## Development Commands

Format Rust code:

```bash
cargo fmt
```

Lint Rust code:

```bash
cargo clippy --all-targets --no-default-features --features pg17
```

Run a local PostgreSQL development instance:

```bash
cargo pgrx run pg17
```

Package the extension artifacts:

```bash
cargo pgrx package --release
```

Install into the configured PostgreSQL installation:

```bash
cargo pgrx install --release
```

## Troubleshooting

### `CREATE EXTENSION pg_embedder` cannot find the extension

Install the extension into the PostgreSQL installation backing your database:

```bash
cargo pgrx install --release
```

Also verify that `cargo pgrx init` was configured with the correct `pg_config` for the PostgreSQL server you are using.

### `embed_ready()` returns `false`

Call:

```sql
SELECT embed_init();
```

Then check:

```sql
SELECT embed_info();
```

Model state is per PostgreSQL backend process, so another connection may not share the same initialized model instance.

### Embeddings are all zeros

This can happen when input text is empty, model initialization failed, or inference failed. Check PostgreSQL logs for warnings emitted by the extension.

### Search quality is poor or results look random

Check which model the running build embeds:

```sql
SELECT embed_model();
```

An English corpus queried against the `model-es` build (or the reverse) produces weak
rankings. Also confirm the stored embeddings were generated by the *same* model as the
query: embeddings from different models are not comparable, so re-embed after switching.

### Dimension mismatch with pgvector

Use `vector(384)` everywhere. The extension currently returns 384-dimensional embeddings:

```sql
SELECT embed_dim();
```

### pgvector casts fail

Prefer the text-vector literal path:

```sql
SELECT embed_text('query')::vector(384);
```

If you use `embed_encode('query')::vector(384)`, ensure your pgvector version supports casts from PostgreSQL real arrays.

### Build fails because PostgreSQL headers are missing

Install the development package for your PostgreSQL version and re-run `cargo pgrx init` with the correct `pg_config` path.

Common package names include `postgresql-server-dev-17`, `postgresql17-devel`, or similar depending on your operating system.

## Version And PostgreSQL Feature Matrix

The default build targets PostgreSQL 17:

```toml
[features]
default = ["pg17", "model-en"]
```

Available feature flags:

| PostgreSQL | Cargo Feature |
| --- | --- |
| 13 | `pg13` |
| 14 | `pg14` |
| 15 | `pg15` |
| 16 | `pg16` |
| 17 | `pg17` |
| 18 | `pg18` |

Build with exactly one PostgreSQL feature at a time, optionally adding a model feature:

```bash
cargo build --release --no-default-features --features pg17
cargo build --release --no-default-features --features pg17,model-es
```

## License

MIT. See `Cargo.toml` for the package license declaration.
