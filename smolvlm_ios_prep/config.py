from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "smolvlm-256m-instruct.json"


@dataclass(frozen=True)
class ArtifactSpec:
    role: str
    source: str
    destination: str
    required: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactSpec":
        return cls(
            role=_required_str(data, "role"),
            source=_required_str(data, "source"),
            destination=_required_str(data, "destination"),
            required=bool(data.get("required", False)),
        )


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str
    revision: str
    target: str
    runtime: dict[str, Any]
    prompting: dict[str, Any]
    artifacts: tuple[ArtifactSpec, ...]

    @classmethod
    def load(cls, path: Path | str | None = None) -> "ModelConfig":
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        artifacts = tuple(ArtifactSpec.from_dict(item) for item in data.get("artifacts", []))
        if not artifacts:
            raise ValueError("Config must define at least one artifact.")
        return cls(
            name=_required_str(data, "name"),
            model_id=_required_str(data, "model_id"),
            revision=str(data.get("revision", "main")),
            target=_required_str(data, "target"),
            runtime=dict(data.get("runtime", {})),
            prompting=dict(data.get("prompting", {})),
            artifacts=artifacts,
        )

    @property
    def required_artifacts(self) -> tuple[ArtifactSpec, ...]:
        return tuple(item for item in self.artifacts if item.required)

    @property
    def optional_artifacts(self) -> tuple[ArtifactSpec, ...]:
        return tuple(item for item in self.artifacts if not item.required)


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Config field '{key}' must be a non-empty string.")
    return value

