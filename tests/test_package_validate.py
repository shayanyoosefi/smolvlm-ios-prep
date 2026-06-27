from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from smolvlm_ios_prep.config import ModelConfig
from smolvlm_ios_prep.package import package_artifacts
from smolvlm_ios_prep.validate import validate_package


class PackageValidateTests(unittest.TestCase):
    def test_package_writes_manifest_and_validate_passes(self) -> None:
        config = ModelConfig.from_dict(
            {
                "name": "test-model",
                "model_id": "example/model",
                "revision": "main",
                "target": "onnx-runtime-ios",
                "artifacts": [
                    {
                        "role": "model",
                        "source": "onnx/model.onnx",
                        "destination": "models/model.onnx",
                        "required": True,
                    },
                    {
                        "role": "optional",
                        "source": "optional.json",
                        "destination": "optional.json",
                        "required": False,
                    },
                ],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            package = root / "package"
            (source / "onnx").mkdir(parents=True)
            (source / "onnx" / "model.onnx").write_bytes(b"fake onnx")

            manifest = package_artifacts(config, source, package)
            self.assertEqual(manifest["missing_required"], [])
            self.assertEqual(manifest["missing_optional"], ["optional.json"])

            manifest_path = package / "manifest.json"
            self.assertTrue(manifest_path.is_file())
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["model_id"], "example/model")

            report = validate_package(package)
            self.assertTrue(report["ok"], report["errors"])

    def test_validate_fails_after_file_changes(self) -> None:
        config = ModelConfig.from_dict(
            {
                "name": "test-model",
                "model_id": "example/model",
                "revision": "main",
                "target": "onnx-runtime-ios",
                "artifacts": [
                    {
                        "role": "tokenizer",
                        "source": "tokenizer.json",
                        "destination": "tokenizer/tokenizer.json",
                        "required": True,
                    }
                ],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            package = root / "package"
            source.mkdir()
            (source / "tokenizer.json").write_text("{}", encoding="utf-8")

            package_artifacts(config, source, package)
            (package / "tokenizer" / "tokenizer.json").write_text('{"changed": true}', encoding="utf-8")

            report = validate_package(package)
            self.assertFalse(report["ok"])
            self.assertTrue(any("Hash mismatch" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()

