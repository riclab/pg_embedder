use once_cell::sync::OnceCell;
use pgrx::prelude::*;
use std::sync::Mutex;

mod engine;
use engine::{EmbedderModel, EMBEDDING_DIM};

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

/// Generate embedding as float array (compatible with pgvector cast).
#[pg_extern(immutable, parallel_safe)]
fn embed_encode(text: &str) -> Vec<f32> {
    if text.is_empty() {
        pgrx::warning!("embed_encode: empty text provided");
        return vec![0.0; EMBEDDING_DIM];
    }

    let model_cell = match MODEL.get() {
        Some(m) => m,
        None => {
            // Auto-initialize if not already done
            if let Ok(model) = EmbedderModel::new() {
                let _ = MODEL.set(Mutex::new(model));
                MODEL.get().unwrap()
            } else {
                pgrx::warning!("Model not initialized. Call embed_init() first.");
                return vec![0.0; EMBEDDING_DIM];
            }
        }
    };

    let model = match model_cell.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };

    match model.encode(text) {
        Ok(embedding) => embedding,
        Err(e) => {
            pgrx::warning!("Encoding failed: {}", e);
            vec![0.0; EMBEDDING_DIM]
        }
    }
}

/// Generate embedding and return as text (for easy casting to vector).
/// Usage: SELECT embed_text('hello')::vector(384)
#[pg_extern(immutable, parallel_safe)]
fn embed_text(text: &str) -> String {
    let embedding = embed_encode(text);
    format!(
        "[{}]",
        embedding
            .iter()
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Batch encode multiple texts. Returns array of embeddings.
#[pg_extern]
fn embed_encode_batch(texts: Vec<String>) -> Vec<Vec<f32>> {
    let model_cell = match MODEL.get() {
        Some(m) => m,
        None => {
            if let Ok(model) = EmbedderModel::new() {
                let _ = MODEL.set(Mutex::new(model));
                MODEL.get().unwrap()
            } else {
                pgrx::warning!("Model not initialized");
                return texts.iter().map(|_| vec![0.0; EMBEDDING_DIM]).collect();
            }
        }
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
        "mxbai-embed-xsmall-v1 | {} dims | status: {} | hybrid mode with pgvector",
        EMBEDDING_DIM, status
    )
}

/// Returns embedding dimension (useful for creating tables).
#[pg_extern(immutable, parallel_safe)]
fn embed_dim() -> i32 {
    EMBEDDING_DIM as i32
}

/// Version information.
#[pg_extern(immutable, parallel_safe)]
fn embed_version() -> &'static str {
    "pg_embedder v0.2.0 (hybrid)"
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
