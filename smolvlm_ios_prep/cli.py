from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import DEFAULT_CONFIG_PATH, ModelConfig
from .contract import ContractError, write_contract
from .download import DownloadError, download_artifacts
from .package import PackageError, package_artifacts
from .validate import validate_package


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except (ContractError, DownloadError, PackageError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help=f"Model config JSON. Default: {DEFAULT_CONFIG_PATH}",
    )

    parser = argparse.ArgumentParser(
        prog="smolvlm-ios-prep",
        description="Prepare SmolVLM artifacts for an iOS app.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Model config JSON. Default: {DEFAULT_CONFIG_PATH}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        parents=[config_parent],
        help="Print the configured artifact plan.",
    )
    inspect_parser.set_defaults(func=_cmd_inspect)

    download_parser = subparsers.add_parser(
        "download",
        parents=[config_parent],
        help="Download configured files from Hugging Face.",
    )
    download_parser.add_argument("--output", type=Path, default=Path("downloads/smolvlm-256m-instruct-q4f16"))
    download_parser.add_argument("--cache-dir", type=Path, default=None)
    download_parser.add_argument("--token", default=None, help="Hugging Face token, if needed.")
    download_parser.set_defaults(func=_cmd_download)

    package_parser = subparsers.add_parser(
        "package",
        parents=[config_parent],
        help="Create the iOS artifact package from a source tree.",
    )
    package_parser.add_argument("--source-dir", type=Path, required=True)
    package_parser.add_argument("--output", type=Path, default=Path("artifacts/ios/smolvlm-256m-instruct-q4f16"))
    package_parser.set_defaults(func=_cmd_package)

    prepare_parser = subparsers.add_parser(
        "prepare",
        parents=[config_parent],
        help="Download and package in one step.",
    )
    prepare_parser.add_argument("--output", type=Path, default=Path("artifacts/ios/smolvlm-256m-instruct-q4f16"))
    prepare_parser.add_argument("--download-dir", type=Path, default=Path("downloads/smolvlm-256m-instruct-q4f16"))
    prepare_parser.add_argument("--cache-dir", type=Path, default=None)
    prepare_parser.add_argument("--token", default=None, help="Hugging Face token, if needed.")
    prepare_parser.set_defaults(func=_cmd_prepare)

    validate_parser = subparsers.add_parser(
        "validate",
        parents=[config_parent],
        help="Validate an iOS artifact package.",
    )
    validate_parser.add_argument("package_dir", type=Path)
    validate_parser.add_argument(
        "--load-onnx",
        action="store_true",
        help="Also try to load ONNX files with ONNX Runtime.",
    )
    validate_parser.add_argument(
        "--onnx-optimization",
        choices=["disabled", "basic", "extended", "all"],
        default="disabled",
        help=(
            "Graph optimization level used for --load-onnx. Defaults to disabled "
            "because some quantized SmolVLM exports trip ONNX Runtime fusions."
        ),
    )
    validate_parser.add_argument("--json", action="store_true", help="Print the full validation report as JSON.")
    validate_parser.set_defaults(func=_cmd_validate)

    contract_parser = subparsers.add_parser(
        "contract",
        help="Generate iOS integration contract files for a package.",
    )
    contract_parser.add_argument("package_dir", type=Path)
    contract_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write ios_contract.json and IOS_INTEGRATION_CONTRACT.md. Default: package_dir.",
    )
    contract_parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Do not load ONNX sessions while generating the contract.",
    )
    contract_parser.add_argument(
        "--onnx-optimization",
        choices=["disabled", "basic", "extended", "all"],
        default="disabled",
        help="Graph optimization level used while collecting ONNX IO metadata.",
    )
    contract_parser.set_defaults(func=_cmd_contract)

    return parser


def _load_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig.load(args.config)


def _cmd_inspect(args: argparse.Namespace) -> int:
    config = _load_config(args)
    payload = {
        "name": config.name,
        "model_id": config.model_id,
        "revision": config.revision,
        "target": config.target,
        "required": [item.source for item in config.required_artifacts],
        "optional": [item.source for item in config.optional_artifacts],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    config = _load_config(args)
    statuses = download_artifacts(
        config,
        args.output,
        cache_dir=args.cache_dir,
        token=args.token,
    )
    _print_status_counts("download", statuses)
    print(f"source_dir={args.output}")
    return 0


def _cmd_package(args: argparse.Namespace) -> int:
    config = _load_config(args)
    manifest = package_artifacts(config, args.source_dir, args.output)
    print(f"package_dir={args.output}")
    print(f"manifest={args.output / 'manifest.json'}")
    if manifest["missing_optional"]:
        print(f"missing_optional={len(manifest['missing_optional'])}")
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    config = _load_config(args)
    statuses = download_artifacts(
        config,
        args.download_dir,
        cache_dir=args.cache_dir,
        token=args.token,
    )
    _print_status_counts("download", statuses)
    manifest = package_artifacts(config, args.download_dir, args.output)
    print(f"package_dir={args.output}")
    print(f"manifest={args.output / 'manifest.json'}")
    if manifest["missing_optional"]:
        print(f"missing_optional={len(manifest['missing_optional'])}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    report = validate_package(
        args.package_dir,
        load_onnx=args.load_onnx,
        onnx_optimization=args.onnx_optimization,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "ok" if report["ok"] else "failed"
        print(f"status={status}")
        print(f"package_dir={report['package_dir']}")
        print(f"checks={len(report['checks'])}")
        if report["errors"]:
            print("errors:")
            for error in report["errors"]:
                print(f"- {error}")
    return 0 if report["ok"] else 1


def _cmd_contract(args: argparse.Namespace) -> int:
    outputs = write_contract(
        args.package_dir,
        args.output_dir,
        load_onnx=not args.skip_onnx,
        onnx_optimization=args.onnx_optimization,
    )
    print(f"contract_json={outputs['json']}")
    print(f"contract_markdown={outputs['markdown']}")
    return 0


def _print_status_counts(label: str, statuses: dict[str, str]) -> None:
    counts: dict[str, int] = {}
    for status in statuses.values():
        counts[status] = counts.get(status, 0) + 1
    joined = " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    print(f"{label}: {joined}")


if __name__ == "__main__":
    raise SystemExit(main())
