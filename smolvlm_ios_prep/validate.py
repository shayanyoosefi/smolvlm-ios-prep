from __future__ import annotations

from pathlib import Path
from typing import Any

from .manifest import load_manifest, sha256_file


class ValidationError(RuntimeError):
    """Raised when artifact validation fails."""


def validate_package(package_dir: Path, *, load_onnx: bool = False) -> dict[str, Any]:
    manifest = load_manifest(package_dir)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    for item in manifest.get("files", []):
        relative_path = item["path"]
        path = package_dir / relative_path
        exists = path.is_file()
        check: dict[str, Any] = {
            "path": relative_path,
            "role": item.get("role"),
            "required": item.get("required", False),
            "exists": exists,
        }

        if not exists:
            if item.get("required", False):
                errors.append(f"Missing required file: {relative_path}")
            checks.append(check)
            continue

        expected_sha = item.get("sha256")
        actual_sha = sha256_file(path)
        check["sha256_ok"] = expected_sha == actual_sha
        if expected_sha != actual_sha:
            errors.append(f"Hash mismatch: {relative_path}")

        expected_size = item.get("bytes")
        actual_size = path.stat().st_size
        check["bytes_ok"] = expected_size == actual_size
        if expected_size != actual_size:
            errors.append(f"Size mismatch: {relative_path}")

        if load_onnx and relative_path.endswith(".onnx"):
            load_result = _load_onnx(path)
            check["onnx_load"] = load_result
            if not load_result["ok"]:
                errors.append(f"ONNX load failed for {relative_path}: {load_result['error']}")

        checks.append(check)

    return {
        "ok": not errors,
        "package_dir": str(package_dir),
        "model_id": manifest.get("model_id"),
        "revision": manifest.get("revision"),
        "checks": checks,
        "errors": errors,
    }


def _load_onnx(path: Path) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        return {
            "ok": False,
            "error": "onnxruntime is not installed. Install with: pip install -e \".[validate]\"",
            "exception": exc.__class__.__name__,
        }

    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # pragma: no cover - depends on optional runtime internals
        return {
            "ok": False,
            "error": str(exc),
            "exception": exc.__class__.__name__,
        }

    return {
        "ok": True,
        "inputs": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in session.get_inputs()
        ],
        "outputs": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in session.get_outputs()
        ],
    }

