from __future__ import annotations

from pathlib import Path
import shutil

from .config import ModelConfig
from .download import required_sources_missing
from .manifest import write_manifest


class PackageError(RuntimeError):
    """Raised when an iOS artifact package cannot be created."""


def package_artifacts(config: ModelConfig, source_dir: Path, output_dir: Path) -> dict:
    missing_required = required_sources_missing(config, source_dir)
    if missing_required:
        missing = ", ".join(item.source for item in missing_required)
        raise PackageError(f"Source directory is missing required file(s): {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for artifact in config.artifacts:
        source = source_dir / artifact.source
        if not source.is_file():
            continue
        destination = output_dir / artifact.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    manifest = write_manifest(config, output_dir)
    _write_artifact_readme(config, output_dir, manifest)
    return manifest


def _write_artifact_readme(config: ModelConfig, output_dir: Path, manifest: dict) -> None:
    required_count = sum(1 for item in manifest["files"] if item["required"] and item["present"])
    optional_count = sum(1 for item in manifest["files"] if not item["required"] and item["present"])
    lines = [
        f"# {config.name} iOS artifacts",
        "",
        f"Model: `{config.model_id}`",
        f"Revision: `{config.revision}`",
        f"Target: `{config.target}`",
        "",
        f"Required files present: {required_count}/{len(config.required_artifacts)}",
        f"Optional files present: {optional_count}/{len(config.optional_artifacts)}",
        "",
        "Read `manifest.json` from the iOS app before opening model files.",
        "It contains file roles, relative paths, SHA-256 hashes, and runtime notes.",
        "",
        "The ONNX decoder loop is app-side work. This package provides the model",
        "components and processing configuration; it does not generate Swift code.",
        "",
    ]
    if manifest["missing_optional"]:
        lines.extend(["Missing optional files:", ""])
        lines.extend(f"- `{item}`" for item in manifest["missing_optional"])
        lines.append("")
    (output_dir / "README.artifacts.md").write_text("\n".join(lines), encoding="utf-8")

