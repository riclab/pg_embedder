-- ============================================================
-- RAG (Retrieval Augmented Generation) Setup
-- ============================================================

-- Document storage for RAG applications
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT NOT NULL,
    source TEXT,
    metadata JSONB DEFAULT '{}',
    embedding vector(384),
    chunk_index INT DEFAULT 0,
    parent_id INT REFERENCES documents(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
    ON documents USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS documents_source_idx ON documents(source);
CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents USING gin(metadata);

-- Auto-embed trigger
CREATE OR REPLACE FUNCTION documents_auto_embed()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.embedding IS NULL THEN
        NEW.embedding := to_embedding(
            COALESCE(NEW.title, '') || ' ' || NEW.content
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS documents_embed_trigger ON documents;
CREATE TRIGGER documents_embed_trigger
    BEFORE INSERT ON documents
    FOR EACH ROW
    EXECUTE FUNCTION documents_auto_embed();

-- ============================================================
-- Chunking Function (for long documents)
-- ============================================================

CREATE OR REPLACE FUNCTION chunk_and_store(
    doc_title TEXT,
    doc_content TEXT,
    doc_source TEXT,
    chunk_size INT DEFAULT 500,
    chunk_overlap INT DEFAULT 50
)
RETURNS INT
LANGUAGE plpgsql
AS $$
DECLARE
    parent_doc_id INT;
    chunk_text TEXT;
    chunk_idx INT := 0;
    content_length INT;
    start_pos INT := 1;
BEGIN
    -- Insert parent document (without embedding)
    INSERT INTO documents (title, content, source, metadata)
    VALUES (doc_title, doc_content, doc_source, jsonb_build_object('type', 'parent'))
    RETURNING id INTO parent_doc_id;
    
    content_length := LENGTH(doc_content);
    
    -- Create chunks
    WHILE start_pos <= content_length LOOP
        chunk_text := SUBSTRING(doc_content FROM start_pos FOR chunk_size);
        
        -- Insert chunk with embedding
        INSERT INTO documents (title, content, source, chunk_index, parent_id, metadata)
        VALUES (
            doc_title,
            chunk_text,
            doc_source,
            chunk_idx,
            parent_doc_id,
            jsonb_build_object('type', 'chunk', 'parent_id', parent_doc_id)
        );
        
        chunk_idx := chunk_idx + 1;
        start_pos := start_pos + chunk_size - chunk_overlap;
    END LOOP;
    
    RETURN parent_doc_id;
END;
$$;

-- ============================================================
-- RAG Search Functions
-- ============================================================

-- Basic RAG retrieval
CREATE OR REPLACE FUNCTION rag_search(
    query TEXT,
    limit_count INT DEFAULT 5
)
RETURNS TABLE(
    id INT,
    title TEXT,
    content TEXT,
    source TEXT,
    similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT 
        d.id,
        d.title,
        d.content,
        d.source,
        1 - (d.embedding <=> to_embedding(query)) AS similarity
    FROM documents d
    WHERE d.parent_id IS NOT NULL  -- Only return chunks
    ORDER BY d.embedding <=> to_embedding(query)
    LIMIT limit_count
$$;

-- RAG search with context window (returns surrounding chunks)
CREATE OR REPLACE FUNCTION rag_search_with_context(
    query TEXT,
    context_window INT DEFAULT 1,
    limit_count INT DEFAULT 3
)
RETURNS TABLE(
    chunk_id INT,
    title TEXT,
    combined_content TEXT,
    source TEXT,
    similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
    WITH ranked_chunks AS (
        SELECT 
            d.id,
            d.title,
            d.content,
            d.source,
            d.parent_id,
            d.chunk_index,
            1 - (d.embedding <=> to_embedding(query)) AS similarity,
            ROW_NUMBER() OVER (ORDER BY d.embedding <=> to_embedding(query)) AS rank
        FROM documents d
        WHERE d.parent_id IS NOT NULL
    ),
    top_chunks AS (
        SELECT * FROM ranked_chunks WHERE rank <= limit_count
    )
    SELECT 
        tc.id AS chunk_id,
        tc.title,
        STRING_AGG(ctx.content, ' ' ORDER BY ctx.chunk_index) AS combined_content,
        tc.source,
        tc.similarity
    FROM top_chunks tc
    LEFT JOIN documents ctx ON ctx.parent_id = tc.parent_id
        AND ctx.chunk_index BETWEEN tc.chunk_index - context_window 
                                AND tc.chunk_index + context_window
    GROUP BY tc.id, tc.title, tc.source, tc.similarity
    ORDER BY tc.similarity DESC
$$;

-- ============================================================
-- Sample Data
-- ============================================================

-- Insert sample documents
INSERT INTO documents (title, content, source) VALUES
('PostgreSQL Introduction', 
 'PostgreSQL is a powerful, open source object-relational database system with over 35 years of active development. It has earned a strong reputation for reliability, feature robustness, and performance.',
 'docs'),
('Vector Search Basics',
 'Vector search enables semantic similarity queries by comparing numerical representations of data. Unlike traditional keyword search, vector search understands the meaning behind queries.',
 'docs'),
('Machine Learning Overview',
 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. Deep learning uses neural networks with multiple layers.',
 'docs'),
('Embedding Models',
 'Embedding models convert text, images, or other data into dense numerical vectors. These vectors capture semantic meaning and enable similarity comparisons.',
 'docs');

-- ============================================================
-- Example Usage
-- ============================================================

\echo ''
\echo '=== RAG Search Examples ==='
\echo ''

\echo 'Query: "how do databases work"'
SELECT * FROM rag_search('how do databases work', 3);

\echo ''
\echo 'Query: "AI and neural networks"'
SELECT * FROM rag_search('AI and neural networks', 3);

\echo ''
\echo 'Query: "converting text to numbers"'
SELECT * FROM rag_search('converting text to numbers', 3);