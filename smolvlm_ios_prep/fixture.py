from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .manifest import sha256_file
from .validate import validate_package


FIXTURE_MANIFEST_NAME = "fixture_manifest.json"


class FixtureError(RuntimeError):
    """Raised when a preprocessing/tokenizer fixture cannot be generated."""


def write_fixture(
    package_dir: Path,
    output_dir: Path,
    *,
    prompt: str,
    image_path: Path | None = None,
    add_generation_prompt: bool = True,
) -> dict[str, Any]:
    validation = validate_package(package_dir, load_onnx=False)
    if not validation["ok"]:
        joined = "; ".join(validation["errors"])
        raise FixtureError(f"Package validation failed: {joined}")

    deps = _load_dependencies()
    np = deps["np"]
    Image = deps["Image"]
    AutoProcessor = deps["AutoProcessor"]

    image = _load_or_create_image(Image, image_path)

    with tempfile.TemporaryDirectory(prefix="smolvlm-hf-files-") as tmp:
        hf_dir = Path(tmp)
        _materialize_hf_processor_dir(package_dir, hf_dir)
        processor = AutoProcessor.from_pretrained(str(hf_dir), local_files_only=True)

        messages = _messages(prompt)
        try:
            rendered_prompt = processor.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
        except ImportError as exc:
            raise FixtureError(
                f"Could not render the chat template: {exc}. "
                'Install fixture dependencies with: pip install -e ".[fixtures]"'
            ) from exc
        inputs = processor(text=rendered_prompt, images=[image], return_tensors="np")

    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_image_path = output_dir / "input_image.png"
    image.save(fixture_image_path)

    tensors = {}
    for name, value in sorted(inputs.items()):
        array = np.asarray(value)
        tensor_path = output_dir / f"{name}.npy"
        np.save(tensor_path, array)
        tensors[name] = _tensor_record(array, tensor_path, output_dir)

    rendered_prompt_path = output_dir / "rendered_prompt.txt"
    rendered_prompt_path.write_text(rendered_prompt, encoding="utf-8")

    input_ids = np.asarray(inputs.get("input_ids", []))
    image_token_id = _image_token_id(package_dir)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": str(package_dir),
        "prompt": prompt,
        "rendered_prompt_path": rendered_prompt_path.name,
        "rendered_prompt": rendered_prompt,
        "add_generation_prompt": add_generation_prompt,
        "image": {
            "path": fixture_image_path.name,
            "source": str(image_path) if image_path else "generated",
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "sha256": sha256_file(fixture_image_path),
        },
        "image_token_id": image_token_id,
        "image_token_positions": _token_positions(input_ids, image_token_id, np),
        "input_ids_preview": input_ids.reshape(-1).astype(int).tolist()[:120],
        "tensors": tensors,
    }

    manifest_path = output_dir / FIXTURE_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_fixture_readme(output_dir)
    return manifest


def _load_dependencies() -> dict[str, Any]:
    missing = [
        package_name
        for module_name, package_name in (
            ("numpy", "numpy"),
            ("PIL", "Pillow"),
            ("transformers", "transformers"),
            ("jinja2", "Jinja2"),
        )
        if importlib.util.find_spec(module_name) is None
    ]

    if missing:
        joined = ", ".join(missing)
        raise FixtureError(
            f"Fixture generation requires missing package(s): {joined}. "
            'Install with: pip install -e ".[fixtures]"'
        )

    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor

    return {"np": np, "Image": Image, "AutoProcessor": AutoProcessor}


def _materialize_hf_processor_dir(package_dir: Path, hf_dir: Path) -> None:
    for subdir in ("processor", "tokenizer"):
        source_dir = package_dir / subdir
        if not source_dir.is_dir():
            continue
        for source in source_dir.iterdir():
            if source.is_file():
                shutil.copy2(source, hf_dir / source.name)


def _load_or_create_image(Image: Any, image_path: Path | None) -> Any:
    if image_path:
        return Image.open(image_path).convert("RGB")

    image = Image.new("RGB", (640, 480), color=(16, 24, 32))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            red = (x * 255) // max(image.width - 1, 1)
            green = (y * 255) // max(image.height - 1, 1)
            blue = 96 if (x // 80 + y // 80) % 2 == 0 else 192
            pixels[x, y] = (red, green, blue)
    return image


def _messages(prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _image_token_id(package_dir: Path) -> int | None:
    config_path = package_dir / "processor" / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    value = config.get("image_token_id")
    return int(value) if value is not None else None


def _token_positions(input_ids: Any, token_id: int | None, np: Any) -> list[list[int]]:
    if token_id is None or getattr(input_ids, "size", 0) == 0:
        return []
    positions = np.argwhere(input_ids == token_id)
    return positions.astype(int).tolist()


def _tensor_record(array: Any, tensor_path: Path, output_dir: Path) -> dict[str, Any]:
    record = {
        "path": str(tensor_path.relative_to(output_dir)),
        "dtype": str(array.dtype),
        "shape": [int(item) for item in array.shape],
        "sha256": sha256_file(tensor_path),
    }
    if array.size:
        if array.dtype.kind in {"b", "i", "u", "f"}:
            record["min"] = _json_number(array.min().item())
            record["max"] = _json_number(array.max().item())
            if array.dtype.kind == "f":
                record["mean"] = _json_number(array.mean().item())
        if array.dtype.kind in {"i", "u"}:
            flat = array.reshape(-1)
            record["preview"] = flat.astype(int).tolist()[:120]
    return record


def _json_number(value: Any) -> int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if hasattr(value, "is_integer") and value.is_integer():
        return int(value)
    return float(value)


def _write_fixture_readme(output_dir: Path) -> None:
    lines = [
        "# SmolVLM iOS Fixture",
        "",
        "This directory is generated by `smolvlm-ios-prep fixture`.",
        "",
        "Use these files to test iOS preprocessing/tokenization parity before",
        "debugging ONNX Runtime decoder behavior.",
        "",
        "- `fixture_manifest.json` describes the prompt, image, tensor files, shapes, dtypes, and hashes.",
        "- `rendered_prompt.txt` is the exact prompt after applying the chat template.",
        "- `*.npy` files are NumPy tensors exported by Hugging Face's local processor.",
        "",
    ]
    (output_dir / "README.fixture.md").write_text("\n".join(lines), encoding="utf-8")
