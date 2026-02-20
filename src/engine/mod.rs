use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use tokenizers::Tokenizer;

// Embedded weights in binary
const MODEL_WEIGHTS: &[u8] = include_bytes!("../../weights/es/model.safetensors");
const TOKENIZER_JSON: &[u8] = include_bytes!("../../weights/es/tokenizer.json");
const CONFIG_JSON: &[u8] = include_bytes!("../../weights/es/config.json");

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

        let pooled = self.mean_pooling(&embeddings, &attention_mask_tensor)?;
        let normalized = self.l2_normalize(&pooled)?;

        normalized
            .squeeze(0)
            .map_err(|e| anyhow!("Squeeze error: {}", e))?
            .to_vec1::<f32>()
            .map_err(|e| anyhow!("Vec conversion error: {}", e))
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
