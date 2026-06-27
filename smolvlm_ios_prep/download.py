from __future__ import annotations

from pathlib import Path
import shutil

from .config import ArtifactSpec, ModelConfig


class DownloadError(RuntimeError):
    """Raised when required model files cannot be downloaded."""


def download_artifacts(
    config: ModelConfig,
    output_dir: Path,
    *,
    cache_dir: Path | None = None,
    token: str | None = None,
) -> dict[str, str]:
    """Download configured Hugging Face files into a source tree.

    Returns a map of source path to status: "downloaded" or "missing_optional".
    """

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
    except ImportError as exc:
        raise DownloadError(
            "Downloading requires huggingface-hub. Install with: "
            'pip install -e ".[download]"'
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = {}

    for artifact in config.artifacts:
        try:
            downloaded = hf_hub_download(
                repo_id=config.model_id,
                filename=artifact.source,
                revision=config.revision,
                cache_dir=str(cache_dir) if cache_dir else None,
                token=token,
            )
        except EntryNotFoundError:
            if artifact.required:
                raise DownloadError(f"Required upstream file is missing: {artifact.source}") from None
            statuses[artifact.source] = "missing_optional"
            continue
        except HfHubHTTPError as exc:
            if artifact.required:
                raise DownloadError(f"Could not download required file {artifact.source}: {exc}") from exc
            statuses[artifact.source] = "missing_optional"
            continue

        destination = output_dir / artifact.source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, destination)
        statuses[artifact.source] = "downloaded"

    _write_download_readme(config, output_dir)
    return statuses


def required_sources_missing(config: ModelConfig, source_dir: Path) -> list[ArtifactSpec]:
    return [artifact for artifact in config.required_artifacts if not (source_dir / artifact.source).is_file()]


def _write_download_readme(config: ModelConfig, output_dir: Path) -> None:
    lines = [
        f"# {config.name} source files",
        "",
        f"Downloaded from `{config.model_id}` revision `{config.revision}`.",
        "",
        "This directory preserves upstream paths. Use `smolvlm-ios-prep package`",
        "to build the iOS artifact layout.",
        "",
    ]
    (output_dir / "README.download.md").write_text("\n".join(lines), encoding="utf-8")

