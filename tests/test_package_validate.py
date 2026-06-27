from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from smolvlm_ios_prep.contract import build_contract, write_contract
from smolvlm_ios_prep.config import ModelConfig
from smolvlm_ios_prep.fixture import _messages, _token_positions
from smolvlm_ios_prep.package import package_artifacts
from smolvlm_ios_prep.validate import _optimization_level, validate_package


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

    def test_optimization_level_rejects_unknown_value(self) -> None:
        class FakeGraphOptimizationLevel:
            ORT_DISABLE_ALL = object()

        class FakeOrt:
            GraphOptimizationLevel = FakeGraphOptimizationLevel

        with self.assertRaises(ValueError):
            _optimization_level(FakeOrt, "surprise")

    def test_contract_generation_without_onnx_load(self) -> None:
        config = ModelConfig.from_dict(
            {
                "name": "test-model",
                "model_id": "example/model",
                "revision": "main",
                "target": "onnx-runtime-ios",
                "artifacts": [
                    {
                        "role": "model_config",
                        "source": "config.json",
                        "destination": "processor/config.json",
                        "required": True,
                    },
                    {
                        "role": "image_processor",
                        "source": "preprocessor_config.json",
                        "destination": "processor/preprocessor_config.json",
                        "required": True,
                    },
                    {
                        "role": "tokenizer_config",
                        "source": "tokenizer_config.json",
                        "destination": "tokenizer/tokenizer_config.json",
                        "required": True,
                    },
                ],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            package = root / "package"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Idefics3ForConditionalGeneration"],
                        "model_type": "idefics3",
                        "image_token_id": 49190,
                        "text_config": {
                            "hidden_size": 576,
                            "vocab_size": 49280,
                            "num_hidden_layers": 30,
                            "num_key_value_heads": 3,
                            "head_dim": 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (source / "preprocessor_config.json").write_text(
                json.dumps({"processor_class": "Idefics3Processor", "image_mean": [0.5, 0.5, 0.5]}),
                encoding="utf-8",
            )
            (source / "tokenizer_config.json").write_text(
                json.dumps({"eos_token": "<|im_end|>"}),
                encoding="utf-8",
            )
            package_artifacts(config, source, package)

            contract = build_contract(package, load_onnx=False)
            self.assertEqual(contract["model"]["image_token_id"], 49190)
            self.assertEqual(contract["processor"]["processor_class"], "Idefics3Processor")

            outputs = write_contract(package, load_onnx=False)
            self.assertTrue(outputs["json"].is_file())
            self.assertTrue(outputs["markdown"].is_file())

    def test_fixture_message_shape(self) -> None:
        messages = _messages("What is shown?")
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"][0]["type"], "image")
        self.assertEqual(messages[0]["content"][1]["text"], "What is shown?")

    def test_token_positions(self) -> None:
        import numpy as np

        positions = _token_positions(np.asarray([[1, 49190, 2, 49190]]), 49190, np)
        self.assertEqual(positions, [[0, 1], [0, 3]])


if __name__ == "__main__":
    unittest.main()
