from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .manifest import load_manifest
from .validate import validate_package


CONTRACT_JSON_NAME = "ios_contract.json"
CONTRACT_MARKDOWN_NAME = "IOS_INTEGRATION_CONTRACT.md"


class ContractError(RuntimeError):
    """Raised when an iOS integration contract cannot be generated."""


def write_contract(
    package_dir: Path,
    output_dir: Path | None = None,
    *,
    load_onnx: bool = True,
    onnx_optimization: str = "disabled",
) -> dict[str, Path]:
    destination_dir = output_dir or package_dir
    destination_dir.mkdir(parents=True, exist_ok=True)

    contract = build_contract(
        package_dir,
        load_onnx=load_onnx,
        onnx_optimization=onnx_optimization,
    )

    json_path = destination_dir / CONTRACT_JSON_NAME
    markdown_path = destination_dir / CONTRACT_MARKDOWN_NAME
    json_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_contract_markdown(contract), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def build_contract(
    package_dir: Path,
    *,
    load_onnx: bool = True,
    onnx_optimization: str = "disabled",
) -> dict[str, Any]:
    manifest = load_manifest(package_dir)
    validation = validate_package(
        package_dir,
        load_onnx=load_onnx,
        onnx_optimization=onnx_optimization,
    )
    if not validation["ok"]:
        joined = "; ".join(validation["errors"])
        raise ContractError(f"Package validation failed: {joined}")

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package": _package_summary(manifest),
        "runtime_requirements": {
            "onnx_graph_optimization": onnx_optimization if load_onnx else "not_checked",
            "session_creation_note": (
                "Use disabled graph optimization for the q4f16 package unless a target "
                "iOS ONNX Runtime build has been tested with a higher level."
            ),
        },
        "files": _files_by_role(manifest),
        "processor": _processor_summary(package_dir),
        "tokenizer": _tokenizer_summary(package_dir),
        "model": _model_summary(package_dir),
        "onnx": _onnx_summary(validation),
        "ios_integration_steps": _integration_steps(),
    }


def render_contract_markdown(contract: dict[str, Any]) -> str:
    package = contract["package"]
    runtime = contract["runtime_requirements"]
    processor = contract["processor"]
    model = contract["model"]
    tokenizer = contract["tokenizer"]

    lines = [
        f"# {package['package_name']} iOS Integration Contract",
        "",
        f"Model: `{package['model_id']}`",
        f"Revision: `{package['revision']}`",
        f"Target: `{package['target']}`",
        f"Generated: `{contract['generated_at']}`",
        "",
        "## Runtime",
        "",
        f"- ONNX graph optimization: `{runtime['onnx_graph_optimization']}`",
        f"- Note: {runtime['session_creation_note']}",
        "",
        "## Files",
        "",
        "| Role | Path | Required | Bytes |",
        "| --- | --- | --- | ---: |",
    ]

    for role, item in sorted(contract["files"].items()):
        lines.append(
            f"| `{role}` | `{item['path']}` | `{str(item['required']).lower()}` | {item.get('bytes', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Image Processor",
            "",
            f"- Processor: `{processor.get('processor_class')}` / `{processor.get('image_processor_type')}`",
            f"- Convert RGB: `{_json_inline(processor.get('do_convert_rgb'))}`",
            f"- Resize: `{_json_inline(processor.get('do_resize'))}` with size `{_json_inline(processor.get('size'))}`",
            f"- Max image size: `{_json_inline(processor.get('max_image_size'))}`",
            f"- Image splitting: `{_json_inline(processor.get('do_image_splitting'))}`",
            f"- Image sequence length: `{processor.get('image_seq_len')}`",
            f"- Rescale factor: `{processor.get('rescale_factor')}`",
            f"- Mean: `{_json_inline(processor.get('image_mean'))}`",
            f"- Std: `{_json_inline(processor.get('image_std'))}`",
            "",
            "## Tokenizer And Model",
            "",
            f"- Model type: `{model.get('model_type')}`",
            f"- Architecture: `{', '.join(model.get('architectures', []))}`",
            f"- Image token id: `{model.get('image_token_id')}`",
            f"- Hidden size: `{model.get('hidden_size')}`",
            f"- Vocab size: `{model.get('vocab_size')}`",
            f"- Decoder layers: `{model.get('num_hidden_layers')}`",
            f"- KV heads: `{model.get('num_key_value_heads')}`",
            f"- KV head dim: `{model.get('head_dim')}`",
            f"- BOS/EOS/PAD: `{tokenizer.get('bos_token')}` / `{tokenizer.get('eos_token')}` / `{tokenizer.get('pad_token')}`",
            f"- Chat template file present: `{_json_inline(tokenizer.get('chat_template_file_present'))}`",
            "",
            "## ONNX Interfaces",
            "",
        ]
    )

    for role, model_io in sorted(contract["onnx"].items()):
        lines.extend(_render_model_io(role, model_io))

    lines.extend(
        [
            "## iOS Integration Steps",
            "",
        ]
    )
    for index, step in enumerate(contract["ios_integration_steps"], start=1):
        lines.append(f"{index}. {step}")
    lines.append("")
    return "\n".join(lines)


def _package_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "package_name": manifest.get("package_name"),
        "model_id": manifest.get("model_id"),
        "revision": manifest.get("revision"),
        "target": manifest.get("target"),
        "manifest_schema_version": manifest.get("schema_version"),
    }


def _files_by_role(manifest: dict[str, Any]) -> dict[str, Any]:
    files = {}
    for item in manifest.get("files", []):
        if not item.get("present"):
            continue
        files[item["role"]] = {
            "path": item["path"],
            "required": item["required"],
            "bytes": item.get("bytes"),
            "sha256": item.get("sha256"),
        }
    return files


def _processor_summary(package_dir: Path) -> dict[str, Any]:
    preprocessor = _read_json_if_present(package_dir / "processor" / "preprocessor_config.json")
    processor = _read_json_if_present(package_dir / "processor" / "processor_config.json")
    return {
        "processor_class": preprocessor.get("processor_class") or processor.get("processor_class"),
        "image_processor_type": preprocessor.get("image_processor_type"),
        "do_convert_rgb": preprocessor.get("do_convert_rgb"),
        "do_image_splitting": preprocessor.get("do_image_splitting"),
        "do_normalize": preprocessor.get("do_normalize"),
        "do_pad": preprocessor.get("do_pad"),
        "do_rescale": preprocessor.get("do_rescale"),
        "do_resize": preprocessor.get("do_resize"),
        "size": preprocessor.get("size"),
        "max_image_size": preprocessor.get("max_image_size"),
        "resample": preprocessor.get("resample"),
        "rescale_factor": preprocessor.get("rescale_factor"),
        "image_mean": preprocessor.get("image_mean"),
        "image_std": preprocessor.get("image_std"),
        "image_seq_len": processor.get("image_seq_len"),
    }


def _tokenizer_summary(package_dir: Path) -> dict[str, Any]:
    config = _read_json_if_present(package_dir / "tokenizer" / "tokenizer_config.json")
    special_tokens = _read_json_if_present(package_dir / "tokenizer" / "special_tokens_map.json")
    return {
        "tokenizer_class": config.get("tokenizer_class"),
        "model_max_length": config.get("model_max_length"),
        "bos_token": _token_content(special_tokens.get("bos_token") or config.get("bos_token")),
        "eos_token": _token_content(special_tokens.get("eos_token") or config.get("eos_token")),
        "pad_token": _token_content(special_tokens.get("pad_token") or config.get("pad_token")),
        "unk_token": _token_content(special_tokens.get("unk_token") or config.get("unk_token")),
        "chat_template_in_tokenizer_config": bool(config.get("chat_template")),
        "chat_template_file_present": (package_dir / "tokenizer" / "chat_template.json").is_file(),
    }


def _model_summary(package_dir: Path) -> dict[str, Any]:
    config = _read_json_if_present(package_dir / "processor" / "config.json")
    text_config = config.get("text_config") or {}
    return {
        "architectures": config.get("architectures", []),
        "model_type": config.get("model_type"),
        "image_token_id": config.get("image_token_id"),
        "scale_factor": config.get("scale_factor"),
        "hidden_size": text_config.get("hidden_size"),
        "vocab_size": text_config.get("vocab_size"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "num_attention_heads": text_config.get("num_attention_heads"),
        "num_key_value_heads": text_config.get("num_key_value_heads"),
        "head_dim": text_config.get("head_dim"),
        "max_position_embeddings": text_config.get("max_position_embeddings"),
        "use_cache": text_config.get("use_cache"),
        "kv_cache_dtype": (text_config.get("transformers.js_config") or {}).get("kv_cache_dtype"),
    }


def _onnx_summary(validation: dict[str, Any]) -> dict[str, Any]:
    models = {}
    for check in validation.get("checks", []):
        onnx_load = check.get("onnx_load")
        if not onnx_load:
            continue
        models[check["role"]] = {
            "path": check["path"],
            "optimization": onnx_load.get("optimization"),
            "inputs": onnx_load.get("inputs", []),
            "outputs": onnx_load.get("outputs", []),
        }
    return models


def _integration_steps() -> list[str]:
    return [
        "Bundle the artifact directory with the app or copy it into app storage unchanged.",
        "Load manifest.json first and resolve all file paths relative to the artifact directory.",
        "Create ONNX Runtime sessions for vision_encoder, token_embedding, and decoder with graph optimizations disabled for the q4f16 package.",
        "Apply processor/preprocessor_config.json image rules before calling vision_encoder.",
        "Tokenize the chat prompt with tokenizer/tokenizer.json and the packaged chat template.",
        "Replace image-token embeddings with vision_encoder image_features before decoder prefill.",
        "Run the decoder autoregressive loop while feeding present.* outputs back as past_key_values.* inputs.",
        "Stop on eos_token or the app's max-new-token limit, then detokenize generated token ids.",
    ]


def _render_model_io(role: str, model_io: dict[str, Any]) -> list[str]:
    lines = [
        f"### `{role}`",
        "",
        f"Path: `{model_io['path']}`",
        "",
        "Inputs:",
        "",
    ]
    for item in _compact_io(model_io.get("inputs", [])):
        lines.append(f"- `{item['name']}` `{item['type']}` shape `{_json_inline(item['shape'])}`")
    lines.extend(["", "Outputs:", ""])
    for item in _compact_io(model_io.get("outputs", [])):
        lines.append(f"- `{item['name']}` `{item['type']}` shape `{_json_inline(item['shape'])}`")
    lines.append("")
    return lines


def _compact_io(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    past = [item for item in items if item["name"].startswith("past_key_values.")]
    present = [item for item in items if item["name"].startswith("present.")]
    regular = [
        item
        for item in items
        if not item["name"].startswith("past_key_values.") and not item["name"].startswith("present.")
    ]
    compacted = list(regular)
    if past:
        compacted.append(_range_summary(past, "past_key_values"))
    if present:
        compacted.append(_range_summary(present, "present"))
    return compacted


def _range_summary(items: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    return {
        "name": f"{prefix}.0-{(len(items) // 2) - 1}.key/value",
        "type": items[0]["type"],
        "shape": items[0]["shape"],
    }


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_inline(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _token_content(value: Any) -> Any:
    if isinstance(value, dict) and "content" in value:
        return value["content"]
    return value
