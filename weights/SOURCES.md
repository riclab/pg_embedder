# Embedded model assets

`src/engine/mod.rs` compiles one of these directories into the extension binary with
`include_bytes!`. Exactly one is linked per build, selected by Cargo feature:

| Directory | Cargo feature | Pooling | Query prefix | Default |
| --- | --- | --- | --- | --- |
| `en/` | `model-en` | mean | no | yes (also used when no model feature is enabled) |
| `es/` | `model-es` | mean | no | no |
| `en-arctic/` | `model-en-arctic` | cls | yes | no |

All three are 384-dimensional BERT encoders over the same 30522-token English WordPiece
vocabulary. `en/` and `es/` ship a byte-identical `tokenizer.json`; `en-arctic/` uses the
same vocabulary with different tokenizer settings.

Pooling and the query prefix are properties of how each model was trained, not options.
They live next to the weights in `src/engine/mod.rs` because getting either wrong
produces embeddings of the right shape and unit norm that simply rank badly.

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

## `en-arctic/` — Snowflake/snowflake-arctic-embed-xs

Retrieval-tuned English model with the same parameter count (22M) and dimension (384) as
the default. Fetched from <https://huggingface.co/Snowflake/snowflake-arctic-embed-xs> at
revision `d8c86521100d3556476a063fc2342036d45c106f`, Apache-2.0.

`config.json` and `tokenizer.json` are upstream bytes. **`model.safetensors` is derived**,
not upstream: the release ships F32 (90.3 MB), which would double this model's share of
the extension binary for nothing, since candle upcasts to F32 at load either way
(`DTYPE = F32`). It was cast to F16 with the script in `tools/`, taking the payload to
45.1 MB — below the 48.9 MB of `en/`.

```
upstream F32   sha256  ee789e0b1d6ecbbd5ce37b474af556cc1a1319cee4417d9e3b11f82e90300706
derived  F16   sha256  1d48394aa42d64dda1e77ab5027403f1a7e4c6b1b458facfc38b3c2da68ebd78
config.json    sha256  d7d071046ab952af96b7abad788db7ab3fc997b465e1b9914ff39707092254ec
tokenizer.json sha256  91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854
```

Reproduce the derived file exactly:

```bash
rev=d8c86521100d3556476a063fc2342036d45c106f
curl -sSL "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/resolve/$rev/model.safetensors" -o /tmp/arctic-f32.safetensors
shasum -a 256 /tmp/arctic-f32.safetensors   # must match the upstream sha above
python3 weights/tools/convert_f32_to_f16.py /tmp/arctic-f32.safetensors weights/en-arctic/model.safetensors
shasum -a 256 weights/en-arctic/model.safetensors
```

Cast error measured over all 22,565,376 values: max absolute 3.008e-03, max relative
4.880e-04 on values above 1e-3. No value came near F16's 65504 ceiling (the largest
magnitude in the file is 8.78), so nothing overflowed. Note `config.json` still declares
`"torch_dtype": "float32"` — that field describes the upstream release and is ignored by
candle, which reads dtypes from the safetensors header.

This model is **asymmetric**: queries must carry the prefix
`Represent this sentence for searching relevant passages: ` and documents must not.
`embed_query()` applies it; `embed_encode()` does not.

## Adding another model

1. Drop `model.safetensors`, `tokenizer.json`, and `config.json` into `weights/<name>/`.
2. Add a `model-<name>` feature in `Cargo.toml` and a matching `weights` module in
   `src/engine/mod.rs` with its `MODEL_NAME`.
3. Set its `POOLING` and `QUERY_PREFIX` from the upstream `1_Pooling/config.json` and
   `config_sentence_transformers.json`. Do not guess: both fail silently.
4. If `hidden_size` is not 384, update `EMBEDDING_DIM` and every `vector(384)` in `sql/`.
   `EmbedderModel::new()` refuses to load a config whose `hidden_size` disagrees with
   `EMBEDDING_DIM`, so a mismatch surfaces as an init error rather than bad embeddings.
5. Record the source and checksums here. If the file is derived rather than upstream,
   record both checksums and the command that reproduces the derivation.
6. Compare it against the current default with `sql/model_eval.sql` before adopting it.
