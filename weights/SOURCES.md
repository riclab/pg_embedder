# Embedded model assets

`src/engine/mod.rs` compiles one of these directories into the extension binary with
`include_bytes!`. Exactly one is linked per build, selected by Cargo feature:

| Directory | Cargo feature | Default |
| --- | --- | --- |
| `en/` | `model-en` | yes (also used when no model feature is enabled) |
| `es/` | `model-es` | no |

Both are 384-dimensional BERT encoders sharing the same 30522-token English WordPiece
tokenizer (`tokenizer.json` is byte-identical between the two directories).

## `en/` — mixedbread-ai/mxbai-embed-xsmall-v1

The official upstream release, unmodified. Fetched from
<https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1> at revision
`e6ac24e5d6efb8782b59de1647b3ececb4ece94e`. The repository was previously published as
`mixedbread-ai/mxbai-embed-mini-v1`, which is why `config.json` still carries
`"_name_or_path": "mixedbread-ai/mxbai-embed-mini-v1"`; that repo id now redirects to
`mxbai-embed-xsmall-v1`.

Weights are F16 in the file and are upcast to F32 at load time by the Candle `VarBuilder`.

```
sha256  8a1a58f701e103c02ad3a519c04293f57e037b7a47aaac184466d6dd3e76708e  en/model.safetensors
sha256  55f755d351fd04b0fef37760e07e195eb47e15f7aed6fc42d9be3dde3d38bca4  en/config.json
sha256  da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0  en/tokenizer.json
```

Re-download and verify:

```bash
base=https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1/resolve/main
for f in model.safetensors config.json tokenizer.json; do
    curl -sSL "$base/$f" -o "weights/en/$f"
done
shasum -a 256 weights/en/*
```

## `es/` — Spanish variant

A BF16 re-save of the same architecture (saved with transformers 4.57.3), kept for
Spanish-language use. It is not an upstream mixedbread release and has no published
checksum to verify against.

```
sha256  ae11ab0b3189552813c3f0c9b21fea0b939ca8e8c8b3aa8a9e19f390aa77fb5e  es/model.safetensors
sha256  5e2059e4cda8deccf6a778332c8149e80214048fc095f0322c8b36f940ed2804  es/config.json
sha256  da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0  es/tokenizer.json
```

## Adding another model

1. Drop `model.safetensors`, `tokenizer.json`, and `config.json` into `weights/<name>/`.
2. Add a `model-<name>` feature in `Cargo.toml` and a matching `weights` module in
   `src/engine/mod.rs` with its `MODEL_NAME`.
3. If `hidden_size` is not 384, update `EMBEDDING_DIM` and every `vector(384)` in `sql/`.
   `EmbedderModel::new()` refuses to load a config whose `hidden_size` disagrees with
   `EMBEDDING_DIM`, so a mismatch surfaces as an init error rather than bad embeddings.
4. Record the source and checksums here.
