use once_cell::sync::OnceCell;
use pgrx::prelude::*;
use std::sync::Mutex;

mod engine;
use engine::{EmbedderModel, Pooling, EMBEDDING_DIM, MODEL_NAME};

pgrx::pg_module_magic!();

static MODEL: OnceCell<Mutex<EmbedderModel>> = OnceCell::new();

/// Initialize the embedding model. Call once per database session.
#[pg_extern]
fn embed_init() -> &'static str {
    match MODEL.get() {
        Some(_) => "Model already initialized",
        None => match EmbedderModel::new() {
            Ok(model) => match MODEL.set(Mutex::new(model)) {
                Ok(_) => "Model initialized successfully",
                Err(_) => "Model was initialized by another connection",
            },
            Err(e) => {
                pgrx::warning!("Failed to initialize model: {}", e);
                "Failed to initialize model - check logs"
            }
        },
    }
}

/// Hand out the process-local model, initializing it on first use.
///
/// Returns `None` when the model cannot be loaded, so callers degrade to a
/// sentinel instead of raising and aborting the surrounding transaction.
fn model() -> Option<&'static Mutex<EmbedderModel>> {
    if let Some(cell) = MODEL.get() {
        return Some(cell);
    }

    match EmbedderModel::new() {
        Ok(loaded) => {
            let _ = MODEL.set(Mutex::new(loaded));
            MODEL.get()
        }
        Err(e) => {
            pgrx::warning!("Model not initialized ({}). Call embed_init() first.", e);
            None
        }
    }
}

fn encode_one(text: &str, as_query: bool) -> Vec<f32> {
    if text.is_empty() {
        pgrx::warning!("embed_encode: empty text provided");
        return vec![0.0; EMBEDDING_DIM];
    }

    let model_cell = match model() {
        Some(cell) => cell,
        None => return vec![0.0; EMBEDDING_DIM],
    };

    let model = match model_cell.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };

    let encoded = if as_query {
        model.encode_query(text)
    } else {
        model.encode(text)
    };

    encoded.unwrap_or_else(|e| {
        pgrx::warning!("Encoding failed: {}", e);
        vec![0.0; EMBEDDING_DIM]
    })
}

/// Generate embedding as float array (compatible with pgvector cast).
///
/// This is the document/indexing side of a retrieval pair. Use `embed_query()`
/// for search queries.
#[pg_extern(immutable, parallel_safe)]
fn embed_encode(text: &str) -> Vec<f32> {
    encode_one(text, false)
}

/// Generate embedding for a search query.
///
/// Asymmetric models prepend an instruction to queries that documents must not
/// carry. On symmetric models this is identical to `embed_encode()`, so search
/// SQL written against `embed_query()` stays correct across a model swap.
#[pg_extern(immutable, parallel_safe)]
fn embed_query(text: &str) -> Vec<f32> {
    encode_one(text, true)
}

/// Query embedding as a pgvector-compatible literal.
#[pg_extern(immutable, parallel_safe)]
fn embed_query_text(text: &str) -> String {
    vector_literal(&embed_query(text))
}

/// Whether this build's model distinguishes queries from documents.
#[pg_extern(immutable, parallel_safe)]
fn embed_is_asymmetric() -> bool {
    engine::QUERY_PREFIX.is_some()
}

/// Format an embedding the way pgvector parses a `vector` literal.
fn vector_literal(embedding: &[f32]) -> String {
    format!(
        "[{}]",
        embedding
            .iter()
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Generate embedding and return as text (for easy casting to vector).
/// Usage: SELECT embed_text('hello')::vector(384)
#[pg_extern(immutable, parallel_safe)]
fn embed_text(text: &str) -> String {
    vector_literal(&embed_encode(text))
}

/// Batch encode multiple texts. Returns array of embeddings.
#[pg_extern]
fn embed_encode_batch(texts: Vec<String>) -> Vec<Vec<f32>> {
    let model_cell = match model() {
        Some(cell) => cell,
        None => return texts.iter().map(|_| vec![0.0; EMBEDDING_DIM]).collect(),
    };

    let model = match model_cell.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };

    texts
        .iter()
        .map(|t| {
            model.encode(t).unwrap_or_else(|e| {
                pgrx::warning!("Encoding failed for text: {}", e);
                vec![0.0; EMBEDDING_DIM]
            })
        })
        .collect()
}

/// Returns model information.
#[pg_extern]
fn embed_info() -> String {
    let status = if MODEL.get().is_some() {
        "loaded"
    } else {
        "not loaded"
    };
    format!(
        "{} | {} dims | pooling: {} | {} | status: {} | hybrid mode with pgvector",
        MODEL_NAME,
        EMBEDDING_DIM,
        pooling_name(),
        if engine::QUERY_PREFIX.is_some() {
            "asymmetric (use embed_query for searches)"
        } else {
            "symmetric"
        },
        status
    )
}

/// Pooling strategy of the compiled-in model, for `embed_info()`.
fn pooling_name() -> &'static str {
    match engine::POOLING {
        Pooling::Mean => "mean",
        Pooling::Cls => "cls",
    }
}

/// Returns the name of the embedding model compiled into this build.
#[pg_extern(immutable, parallel_safe)]
fn embed_model() -> &'static str {
    MODEL_NAME
}

/// Returns embedding dimension (useful for creating tables).
#[pg_extern(immutable, parallel_safe)]
fn embed_dim() -> i32 {
    EMBEDDING_DIM as i32
}

/// Version information.
#[pg_extern(immutable, parallel_safe)]
fn embed_version() -> &'static str {
    concat!("pg_embedder v", env!("CARGO_PKG_VERSION"), " (hybrid)")
}

/// Check if model is ready.
#[pg_extern]
fn embed_ready() -> bool {
    MODEL.get().is_some()
}

// ============================================================
// Similarity Functions (works without pgvector)
// ============================================================

/// Cosine similarity between two vectors. Returns value in [-1, 1], higher = more similar.
#[pg_extern(immutable, parallel_safe, strict)]
fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        pgrx::warning!("cosine_similarity: vectors must have same non-zero length");
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Cosine distance (1 - similarity). Returns value in [0, 2], lower = more similar.
#[pg_extern(immutable, parallel_safe, strict)]
fn cosine_distance(a: Vec<f32>, b: Vec<f32>) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// L2 (Euclidean) distance. Lower = more similar.
#[pg_extern(immutable, parallel_safe, strict)]
fn l2_distance(a: Vec<f32>, b: Vec<f32>) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        pgrx::warning!("l2_distance: vectors must have same non-zero length");
        return f32::MAX;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Dot product (inner product). For normalized vectors, higher = more similar.
#[pg_extern(immutable, parallel_safe, strict)]
fn dot_product(a: Vec<f32>, b: Vec<f32>) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        pgrx::warning!("dot_product: vectors must have same non-zero length");
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compare two texts directly using cosine similarity.
#[pg_extern(immutable, parallel_safe)]
fn text_similarity(text1: &str, text2: &str) -> f32 {
    let emb1 = embed_encode(text1);
    let emb2 = embed_encode(text2);
    cosine_similarity(emb1, emb2)
}

// ============================================================
// Tests
// ============================================================

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_version() {
        assert!(crate::embed_version().contains("pg_embedder"));
    }

    #[pg_test]
    fn test_dimension() {
        assert_eq!(crate::embed_dim(), 384);
    }

    #[pg_test]
    fn test_model_reported_in_info() {
        assert!(crate::embed_info().contains(crate::embed_model()));
    }

    /// Search SQL is written against embed_query() regardless of model. On a
    /// symmetric build it must stay byte-identical to embed_encode(), or callers
    /// would get different vectors for the same text depending on entry point.
    #[cfg(not(feature = "model-en-arctic"))]
    #[pg_test]
    fn test_query_equals_encode_when_symmetric() {
        crate::embed_init();
        assert!(!crate::embed_is_asymmetric());
        assert_eq!(
            crate::embed_encode("vector search"),
            crate::embed_query("vector search")
        );
    }

    #[cfg(feature = "model-en-arctic")]
    #[pg_test]
    fn test_query_prefix_applied_when_asymmetric() {
        crate::embed_init();
        assert!(crate::embed_is_asymmetric());
        assert!(crate::embed_model().contains("arctic"));

        // The prefix has to actually reach the encoder, otherwise the model is
        // being used symmetrically and silently losing retrieval quality.
        assert_ne!(
            crate::embed_encode("vector search"),
            crate::embed_query("vector search")
        );
    }

    /// End-to-end retrieval sanity, using each side of the pair as intended.
    /// Holds for symmetric and asymmetric builds alike.
    #[cfg(not(feature = "model-es"))]
    #[pg_test]
    fn test_query_retrieves_relevant_document() {
        crate::embed_init();

        let query = crate::embed_query("how do I store embeddings in postgres");
        let relevant = crate::embed_encode("PostgreSQL can store vector embeddings in a table column.");
        let irrelevant = crate::embed_encode("Sourdough needs a starter and a long cold ferment.");

        let hit = crate::cosine_similarity(query.clone(), relevant);
        let miss = crate::cosine_similarity(query, irrelevant);
        assert!(hit > miss, "relevant doc scored {hit}, irrelevant scored {miss}");
    }

    /// The default build embeds the English model, so English near-synonyms must
    /// rank above an unrelated English word. Guards against shipping the wrong
    /// weights under the default feature set.
    #[cfg(not(feature = "model-es"))]
    #[pg_test]
    fn test_default_model_is_english() {
        crate::embed_init();
        assert!(crate::embed_model().contains("(en)"));

        let related = crate::text_similarity("cat", "kitten");
        let unrelated = crate::text_similarity("cat", "bicycle");
        assert!(
            related > unrelated,
            "expected cat/kitten ({related}) to outrank cat/bicycle ({unrelated})"
        );
    }

    #[pg_test]
    fn test_init_and_encode() {
        crate::embed_init();
        assert!(crate::embed_ready());

        let embedding = crate::embed_encode("test");
        assert_eq!(embedding.len(), 384);

        // Check normalization (L2 norm should be ~1.0)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[pg_test]
    fn test_embed_text_format() {
        crate::embed_init();
        let result = crate::embed_text("hello");
        assert!(result.starts_with('['));
        assert!(result.ends_with(']'));
    }

    #[pg_test]
    fn test_empty_text() {
        crate::embed_init();
        let embedding = crate::embed_encode("");
        assert_eq!(embedding.len(), 384);
        assert!(embedding.iter().all(|&x| x == 0.0));
    }

    #[pg_test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((crate::cosine_similarity(a, b) - 1.0).abs() < 0.001);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(crate::cosine_similarity(a, b).abs() < 0.001);
    }

    #[pg_test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        assert!((crate::l2_distance(a, b) - 5.0).abs() < 0.001);
    }

    #[pg_test]
    fn test_text_similarity() {
        crate::embed_init();
        let sim = crate::text_similarity("cat", "cat");
        assert!(sim > 0.99);
    }
}

#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}
    #[must_use]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}
