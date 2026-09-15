"""Model-path configuration tests with external packages stubbed out."""

import importlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


class ModelPathTests(unittest.TestCase):
    def setUp(self):
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = Mock(return_value="cached-gemma.gguf")
        llama = types.ModuleType("llama_cpp")
        llama.Llama = Mock()
        transformers = types.ModuleType("sentence_transformers")
        transformers.SentenceTransformer = Mock()
        transformers.CrossEncoder = Mock()
        vec = types.ModuleType("sqlite_vec")
        vec.load = Mock()

        self.hub = hub
        sys.modules.pop("app.engine", None)
        with (
            patch.dict(
                sys.modules,
                {
                    "huggingface_hub": hub,
                    "llama_cpp": llama,
                    "sentence_transformers": transformers,
                    "sqlite_vec": vec,
                },
            ),
            patch.dict(os.environ, {"MODEL_PATH": ""}),
        ):
            self.engine = importlib.import_module("app.engine")

    def tearDown(self):
        sys.modules.pop("app.engine", None)

    def test_default_download_uses_an_actual_gguf_filename(self):
        self.hub.hf_hub_download.assert_called_once_with(
            repo_id="unsloth/gemma-3-1b-it-GGUF",
            filename="gemma-3-1b-it-Q4_K_M.gguf",
            local_dir=str(self.engine.MODEL_DIR),
        )

    def test_existing_local_model_avoids_download(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "alternative.gguf"
            model.touch()
            self.hub.hf_hub_download.reset_mock()
            with patch.dict(os.environ, {"MODEL_PATH": str(model)}):
                self.assertEqual(self.engine.resolve_model_path(), str(model))
            self.hub.hf_hub_download.assert_not_called()

    def test_missing_local_model_reports_a_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.gguf"
            with patch.dict(os.environ, {"MODEL_PATH": str(missing)}):
                with self.assertRaisesRegex(FileNotFoundError, "MODEL_PATH"):
                    self.engine.resolve_model_path()


if __name__ == "__main__":
    unittest.main()
