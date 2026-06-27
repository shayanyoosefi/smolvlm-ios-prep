# smolvlm-ios-prep

Prepare SmolVLM artifacts for an iOS app.

The first supported target is the official Hugging Face ONNX export for
`HuggingFaceTB/SmolVLM-256M-Instruct`. The default config uses the smaller
`q4f16` ONNX files because that is the realistic first target for an iPhone.
The repo packages the files an iOS app needs into one predictable folder,
writes a manifest with hashes and runtime metadata, and validates that the
package is internally consistent.

Core ML conversion is intentionally a second track. SmolVLM is not a single
plain image classifier graph; the iOS app needs a vision encoder, token
embedding model, decoder loop, tokenizer, image processor settings, and prompt
template. The ONNX package makes that boundary explicit first.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[download,validate]"

smolvlm-ios-prep inspect
smolvlm-ios-prep prepare --output artifacts/ios/smolvlm-256m-instruct-q4f16
smolvlm-ios-prep validate artifacts/ios/smolvlm-256m-instruct-q4f16 --load-onnx
```

If the model is already downloaded locally, skip network access:

```bash
smolvlm-ios-prep package \
  --source-dir /path/to/HuggingFaceTB/SmolVLM-256M-Instruct \
  --output artifacts/ios/smolvlm-256m-instruct
```

## Output Layout

```text
artifacts/ios/smolvlm-256m-instruct-q4f16/
  manifest.json
  README.artifacts.md
  models/
    vision_encoder.onnx
    embed_tokens.onnx
    decoder_model_merged.onnx
  tokenizer/
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json           # included when present upstream
  processor/
    config.json
    generation_config.json
    preprocessor_config.json
    processor_config.json             # included when present upstream
```

The iOS app should read `manifest.json` first. It records the model id,
revision, expected runtime, prompt template notes, file roles, SHA-256 hashes,
and missing optional files.

## Commands

```bash
smolvlm-ios-prep inspect
```

Prints the configured Hugging Face repo and expected files.

```bash
smolvlm-ios-prep download --output downloads/smolvlm-256m-instruct-q4f16
```

Downloads configured files from Hugging Face into a local source tree. Requires
`pip install -e ".[download]"`.

```bash
smolvlm-ios-prep package \
  --source-dir downloads/smolvlm-256m-instruct-q4f16 \
  --output artifacts/ios/smolvlm-256m-instruct-q4f16
```

Copies files from a source tree into the iOS package and writes the manifest.

```bash
smolvlm-ios-prep prepare --output artifacts/ios/smolvlm-256m-instruct-q4f16
```

Downloads and packages in one step.

```bash
smolvlm-ios-prep validate artifacts/ios/smolvlm-256m-instruct-q4f16 --load-onnx
```

Checks required files, verifies manifest hashes, and optionally tries to load
each ONNX model with ONNX Runtime. `--load-onnx` defaults to disabled graph
optimization because some quantized SmolVLM exports can hit ONNX Runtime fusion
bugs during session creation. Use `--onnx-optimization all` only when you want
to test the exact default optimized ONNX Runtime path. Requires
`pip install -e ".[validate]"` for `--load-onnx`.

## iOS Runtime Assumptions

- Start with ONNX Runtime Mobile or full ONNX Runtime for iOS.
- For the default `q4f16` package, start session creation with graph
  optimizations disabled. Re-enable optimizations model by model only after
  testing on the target ONNX Runtime build.
- The app owns the autoregressive decode loop and KV-cache feeding.
- The app must apply the tokenizer chat template and image preprocessing values
  from the packaged processor/tokenizer files.
- Core ML conversion should be treated as a separate optimization milestone
  after the ONNX package is running end to end in the app.

## Notes

- Generated artifacts are ignored by Git.
- The default config lives in `configs/smolvlm-256m-instruct.json` and targets
  the `q4f16` ONNX files.
- Use `--config configs/smolvlm-256m-instruct-fp32.json` when you explicitly
  need the larger full-precision files for parity testing.
- Optional upstream files are included when present and recorded as missing
  when absent.
