-- ============================================================
-- pg_embedder Examples (Pure Rust - No pgvector needed)
-- ============================================================

-- Products table with embedding as float array
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    price DECIMAL(10,2),
    category TEXT,
    embedding FLOAT4[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Auto-embed trigger
CREATE OR REPLACE FUNCTION products_auto_embed()
RETURNS TRIGGER AS $$
BEGIN
    NEW.embedding := embed_encode(
        COALESCE(NEW.name, '') || ' ' || COALESCE(NEW.description, '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS products_embed_trigger ON products;
CREATE TRIGGER products_embed_trigger
    BEFORE INSERT OR UPDATE OF name, description ON products
    FOR EACH ROW
    EXECUTE FUNCTION products_auto_embed();

-- Sample data
INSERT INTO products (name, description, price, category) VALUES
    ('Wireless Bluetooth Headphones', 'Premium noise-cancelling headphones with 30-hour battery', 149.99, 'Electronics'),
    ('Organic Green Tea', 'Japanese matcha green tea, rich in antioxidants', 24.99, 'Food'),
    ('Ergonomic Office Chair', 'Adjustable lumbar support, breathable mesh back', 299.99, 'Furniture'),
    ('Stainless Steel Water Bottle', 'Double-wall insulated, keeps drinks cold 24hrs', 34.99, 'Kitchen'),
    ('Yoga Mat Premium', 'Extra thick 6mm, non-slip surface, eco-friendly', 45.99, 'Sports'),
    ('Mechanical Keyboard', 'RGB backlit, Cherry MX switches, programmable', 129.99, 'Electronics'),
    ('French Press Coffee Maker', 'Borosilicate glass carafe, stainless steel filter', 29.99, 'Kitchen'),
    ('Running Shoes Ultralight', 'Breathable mesh, responsive cushioning', 119.99, 'Sports'),
    ('Smart Watch Fitness', 'Heart rate monitor, GPS tracking, sleep analysis', 199.99, 'Electronics'),
    ('Bamboo Cutting Board Set', 'Set of 3 sizes, antimicrobial surface', 39.99, 'Kitchen')
ON CONFLICT DO NOTHING;

-- ============================================================
-- Search Functions
-- ============================================================

-- Semantic search using cosine similarity
CREATE OR REPLACE FUNCTION search_products(
    query TEXT,
    limit_count INT DEFAULT 5
)
RETURNS TABLE(
    id INT,
    name TEXT,
    description TEXT,
    price DECIMAL,
    similarity FLOAT4
)
LANGUAGE SQL STABLE
AS $$
    WITH query_embedding AS (
        SELECT embed_encode(query) AS emb
    )
    SELECT 
        p.id,
        p.name,
        p.description,
        p.price,
        cosine_similarity(p.embedding, q.emb) AS similarity
    FROM products p, query_embedding q
    ORDER BY similarity DESC
    LIMIT limit_count
$$;

-- Search with category filter
CREATE OR REPLACE FUNCTION search_products_by_category(
    query TEXT,
    cat TEXT,
    limit_count INT DEFAULT 5
)
RETURNS TABLE(
    id INT,
    name TEXT,
    price DECIMAL,
    similarity FLOAT4
)
LANGUAGE SQL STABLE
AS $$
    WITH query_embedding AS (
        SELECT embed_encode(query) AS emb
    )
    SELECT 
        p.id,
        p.name,
        p.price,
        cosine_similarity(p.embedding, q.emb) AS similarity
    FROM products p, query_embedding q
    WHERE p.category = cat
    ORDER BY similarity DESC
    LIMIT limit_count
$$;

-- ============================================================
-- Example Queries
-- ============================================================

\echo ''
\echo '=== Semantic Search Examples ==='

\echo ''
\echo 'Query: "audio equipment for music"'
SELECT * FROM search_products('audio equipment for music', 3);

\echo ''
\echo 'Query: "healthy drinks"'
SELECT * FROM search_products('healthy drinks', 3);

\echo ''
\echo 'Query: "work from home setup"'
SELECT * FROM search_products('work from home setup', 3);

\echo ''
\echo 'Query in Electronics: "portable device"'
SELECT * FROM search_products_by_category('portable device', 'Electronics', 3);

\echo ''
\echo 'Direct text comparison:'
SELECT 
    text_similarity('wireless headphones', 'bluetooth earbuds') AS headphones_earbuds,
    text_similarity('wireless headphones', 'coffee maker') AS headphones_coffee;