use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use tokenizers::Tokenizer;

#[cfg(any(
    all(feature = "model-en", feature = "model-es"),
    all(feature = "model-en", feature = "model-en-arctic"),
    all(feature = "model-es", feature = "model-en-arctic"),
))]
compile_error!(
    "enable exactly one embedding model: `model-en` (default), `model-es`, or \
     `model-en-arctic`. The non-default ones need --no-default-features, e.g. \
     --no-default-features --features pgNN,model-en-arctic"
);

/// How token embeddings are reduced to a single vector.
///
/// This is a property of how the model was trained, not a tuning knob. Using the
/// wrong one yields embeddings that look fine — right shape, unit norm — and rank
/// badly, with nothing raised anywhere.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pooling {
    Mean,
    Cls,
}

// Embedded weights in binary. English (the official mixedbread-ai/mxbai-embed-xsmall-v1
// release) is the default and the fallback when no model feature is selected, so
// --no-default-features builds still link a model.
#[cfg(not(any(feature = "model-es", feature = "model-en-arctic")))]
mod weights {
    use super::Pooling;
    pub const MODEL_WEIGHTS: &[u8] = include_bytes!("../../weights/en/model.safetensors");
    pub const TOKENIZER_JSON: &[u8] = include_bytes!("../../weights/en/tokenizer.json");
    pub const CONFIG_JSON: &[u8] = include_bytes!("../../weights/en/config.json");
    pub const MODEL_NAME: &str = "mxbai-embed-xsmall-v1 (en)";
    pub const POOLING: Pooling = Pooling::Mean;
    pub const QUERY_PREFIX: Option<&str> = None;
}

#[cfg(feature = "model-es")]
mod weights {
    use super::Pooling;
    pub const MODEL_WEIGHTS: &[u8] = include_bytes!("../../weights/es/model.safetensors");
    pub const TOKENIZER_JSON: &[u8] = include_bytes!("../../weights/es/tokenizer.json");
    pub const CONFIG_JSON: &[u8] = include_bytes!("../../weights/es/config.json");
    pub const MODEL_NAME: &str = "mxbai-embed-xsmall-v1-es (es)";
    pub const POOLING: Pooling = Pooling::Mean;
    pub const QUERY_PREFIX: Option<&str> = None;
}

// Spike: same parameter count and dimension as the default, ~4 MB smaller once cast
// to F16, and trained for retrieval. Asymmetric — queries carry a prefix that
// documents must not have. See weights/SOURCES.md.
#[cfg(feature = "model-en-arctic")]
mod weights {
    use super::Pooling;
    pub const MODEL_WEIGHTS: &[u8] = include_bytes!("../../weights/en-arctic/model.safetensors");
    pub const TOKENIZER_JSON: &[u8] = include_bytes!("../../weights/en-arctic/tokenizer.json");
    pub const CONFIG_JSON: &[u8] = include_bytes!("../../weights/en-arctic/config.json");
    pub const MODEL_NAME: &str = "snowflake-arctic-embed-xs (en)";
    pub const POOLING: Pooling = Pooling::Cls;
    pub const QUERY_PREFIX: Option<&str> =
        Some("Represent this sentence for searching relevant passages: ");
}

use weights::{CONFIG_JSON, MODEL_WEIGHTS, TOKENIZER_JSON};

/// Name of the model compiled into this build, as reported by `embed_info()`.
pub const MODEL_NAME: &str = weights::MODEL_NAME;

/// Pooling strategy this model was trained with.
pub const POOLING: Pooling = weights::POOLING;

/// Prefix this model expects on the query side of a retrieval pair, if it is
/// asymmetric. `None` means queries and documents are encoded identically.
pub const QUERY_PREFIX: Option<&str> = weights::QUERY_PREFIX;

pub const EMBEDDING_DIM: usize = 384;
pub const MAX_SEQUENCE_LENGTH: usize = 512;

pub struct EmbedderModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbedderModel {
    pub fn new() -> Result<Self> {
        Self::configure_threading();
        let device = Device::Cpu;

        let config: BertConfig = serde_json::from_slice(CONFIG_JSON)
            .map_err(|e| anyhow!("Failed to parse config: {}", e))?;

        if config.hidden_size != EMBEDDING_DIM {
            return Err(anyhow!(
                "{} has hidden_size {}, but EMBEDDING_DIM is {}; update EMBEDDING_DIM and every vector({}) in sql/",
                MODEL_NAME,
                config.hidden_size,
                EMBEDDING_DIM,
                EMBEDDING_DIM
            ));
        }

        let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        let vb = VarBuilder::from_buffered_safetensors(MODEL_WEIGHTS.to_vec(), DTYPE, &device)
            .map_err(|e| anyhow!("Failed to load weights: {}", e))?;

        let model =
            BertModel::load(vb, &config).map_err(|e| anyhow!("Failed to build model: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn configure_threading() {
        std::env::set_var("OMP_NUM_THREADS", "1");
        std::env::set_var("RAYON_NUM_THREADS", "1");
    }

    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(anyhow!("Input text cannot be empty"));
        }

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let token_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let len = token_ids.len().min(MAX_SEQUENCE_LENGTH);

        let token_ids = &token_ids[..len];
        let attention_mask = &attention_mask[..len];

        let token_ids_tensor = Tensor::new(token_ids, &self.device)
            .map_err(|e| anyhow!("Token tensor error: {}", e))?
            .reshape((1, len))
            .map_err(|e| anyhow!("Reshape error: {}", e))?;

        let attention_mask_tensor = Tensor::new(attention_mask, &self.device)
            .map_err(|e| anyhow!("Mask tensor error: {}", e))?
            .reshape((1, len))
            .map_err(|e| anyhow!("Mask reshape error: {}", e))?
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("Dtype error: {}", e))?;

        let token_type_ids = Tensor::zeros_like(&token_ids_tensor)
            .map_err(|e| anyhow!("Token type error: {}", e))?;

        let embeddings = self
            .model
            .forward(
                &token_ids_tensor,
                &token_type_ids,
                Some(&attention_mask_tensor),
            )
            .map_err(|e| anyhow!("Forward pass error: {}", e))?;

        let pooled = match POOLING {
            Pooling::Mean => self.mean_pooling(&embeddings, &attention_mask_tensor)?,
            Pooling::Cls => self.cls_pooling(&embeddings)?,
        };
        let normalized = self.l2_normalize(&pooled)?;

        normalized
            .squeeze(0)
            .map_err(|e| anyhow!("Squeeze error: {}", e))?
            .to_vec1::<f32>()
            .map_err(|e| anyhow!("Vec conversion error: {}", e))
    }

    /// Encode text as a search query.
    ///
    /// Asymmetric models are trained with an instruction on the query side only;
    /// applying it to documents degrades retrieval. On symmetric models this is
    /// exactly `encode`, so callers can use it unconditionally and stay correct
    /// across a model swap.
    pub fn encode_query(&self, text: &str) -> Result<Vec<f32>> {
        match QUERY_PREFIX {
            Some(prefix) => self.encode(&format!("{prefix}{text}")),
            None => self.encode(text),
        }
    }

    /// Take the [CLS] token's hidden state as the sequence embedding.
    fn cls_pooling(&self, embeddings: &Tensor) -> Result<Tensor> {
        embeddings
            .narrow(1, 0, 1)
            .map_err(|e| anyhow!("CLS narrow error: {}", e))?
            .squeeze(1)
            .map_err(|e| anyhow!("CLS squeeze error: {}", e))
    }

    fn mean_pooling(&self, embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mask_expanded = attention_mask
            .unsqueeze(2)
            .map_err(|e| anyhow!("Unsqueeze error: {}", e))?
            .broadcast_as(embeddings.shape())
            .map_err(|e| anyhow!("Broadcast error: {}", e))?
            .to_dtype(embeddings.dtype())
            .map_err(|e| anyhow!("Dtype error: {}", e))?;

        let sum_embeddings = embeddings
            .mul(&mask_expanded)
            .map_err(|e| anyhow!("Mul error: {}", e))?
            .sum(1)
            .map_err(|e| anyhow!("Sum error: {}", e))?;

        let sum_mask = mask_expanded
            .sum(1)
            .map_err(|e| anyhow!("Mask sum error: {}", e))?
            .clamp(1e-9, f64::MAX)
            .map_err(|e| anyhow!("Clamp error: {}", e))?;

        sum_embeddings
            .div(&sum_mask)
            .map_err(|e| anyhow!("Div error: {}", e))
    }

    fn l2_normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        let norm = tensor
            .sqr()
            .map_err(|e| anyhow!("Sqr error: {}", e))?
            .sum_keepdim(1)
            .map_err(|e| anyhow!("Sum error: {}", e))?
            .sqrt()
            .map_err(|e| anyhow!("Sqrt error: {}", e))?;

        tensor
            .broadcast_div(&norm)
            .map_err(|e| anyhow!("Normalize error: {}", e))
    }
}
