from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import ArtifactSpec, ModelConfig


MANIFEST_NAME = "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(artifact: ArtifactSpec, package_dir: Path, *, present: bool) -> dict[str, Any]:
    destination = package_dir / artifact.destination
    record: dict[str, Any] = {
        "role": artifact.role,
        "source": artifact.source,
        "path": artifact.destination,
        "required": artifact.required,
        "present": present,
    }
    if present:
        record["bytes"] = destination.stat().st_size
        record["sha256"] = sha256_file(destination)
    return record


def build_manifest(config: ModelConfig, package_dir: Path) -> dict[str, Any]:
    files = []
    for artifact in config.artifacts:
        destination = package_dir / artifact.destination
        files.append(artifact_record(artifact, package_dir, present=destination.is_file()))

    missing_required = [item["path"] for item in files if item["required"] and not item["present"]]
    missing_optional = [item["path"] for item in files if not item["required"] and not item["present"]]

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_name": config.name,
        "model_id": config.model_id,
        "revision": config.revision,
        "target": config.target,
        "runtime": config.runtime,
        "prompting": config.prompting,
        "files": files,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }


def write_manifest(config: ModelConfig, package_dir: Path) -> dict[str, Any]:
    manifest = build_manifest(config, package_dir)
    manifest_path = package_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_manifest(package_dir: Path) -> dict[str, Any]:
    manifest_path = package_dir / MANIFEST_NAME
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

