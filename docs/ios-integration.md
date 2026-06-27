# iOS Integration Notes

This repo prepares the model package. The iOS app still owns image
preprocessing, prompt assembly, tokenization, embedding merge, and the decoder
loop.

## Files To Bring Into The App

After running:

```bash
smolvlm-ios-prep prepare --output artifacts/ios/smolvlm-256m-instruct-q4f16
smolvlm-ios-prep contract artifacts/ios/smolvlm-256m-instruct-q4f16
smolvlm-ios-prep fixture artifacts/ios/smolvlm-256m-instruct-q4f16
```

copy or bundle the whole artifact directory unchanged:

```text
smolvlm-256m-instruct-q4f16/
  manifest.json
  ios_contract.json
  IOS_INTEGRATION_CONTRACT.md
  models/
  processor/
  tokenizer/
```

Do not flatten the directories. `manifest.json` and `ios_contract.json` both
use relative paths.

## Swift Loader Template

Add `templates/ios/SmolVLMArtifactManifest.swift` to the iOS app. It decodes
`manifest.json` and resolves model/config paths by role:

```swift
let rootURL = Bundle.main.url(
    forResource: "smolvlm-256m-instruct-q4f16",
    withExtension: nil
)!

let package = try SmolVLMPackage.load(from: rootURL)
let visionURL = try package.visionEncoderURL
let embedURL = try package.tokenEmbeddingURL
let decoderURL = try package.decoderURL
```

## ONNX Runtime Session Settings

For the default `q4f16` package, create sessions with graph optimization
disabled first. In ONNX Runtime terms, this is `ORT_DISABLE_ALL` or the Swift
binding's equivalent `disableAll` setting.

The desktop validator uses this setting because the `q4f16` vision encoder can
fail during ONNX Runtime graph fusion even though the model opens correctly when
optimizations are disabled.

Create three sessions:

```text
vision_encoder  -> models/vision_encoder.onnx
token_embedding -> models/embed_tokens.onnx
decoder         -> models/decoder_model_merged.onnx
```

## Runtime Sequence

1. Decode and resize the image using `processor/preprocessor_config.json`.
2. Build `pixel_values` and `pixel_attention_mask`.
3. Run `vision_encoder` to get `image_features`.
4. Build the chat prompt using `tokenizer/tokenizer.json`,
   `tokenizer/tokenizer_config.json`, and `tokenizer/chat_template.json` when
   present.
5. Run `token_embedding` on `input_ids`.
6. Replace image-token embedding positions with `image_features`.
7. Run decoder prefill with `inputs_embeds`, `attention_mask`,
   `position_ids`, and empty KV cache tensors.
8. For each next token, feed `present.*` back into the matching
   `past_key_values.*` decoder inputs.
9. Stop at the EOS token or the app's max-new-token limit.
10. Detokenize generated token ids.

## Contract Files

Use `ios_contract.json` for machine-readable app integration details. Use
`IOS_INTEGRATION_CONTRACT.md` when implementing or reviewing the runtime. The
generated Markdown summarizes:

- model file roles and sizes
- image preprocessing settings
- tokenizer/model constants
- ONNX input and output names, shapes, and dtypes
- decoder KV-cache interface

## Preprocessing Fixture

The `fixture` command writes Hugging Face-generated reference tensors under
`validation/fixtures/smolvlm-256m-instruct-q4f16/` by default. Use these files
as the first iOS unit test target:

```text
fixture_manifest.json
rendered_prompt.txt
input_ids.npy
attention_mask.npy
pixel_values.npy
pixel_attention_mask.npy
```

The Swift runtime should match the prompt, token ids, image-token positions,
pixel tensor shape, and pixel attention mask before the app tries to run the
decoder loop.
