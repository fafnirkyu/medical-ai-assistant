"""Offline API tests: no model download or vector database required."""

import importlib
import sys
import types
import unittest
from unittest.mock import Mock, patch


class AskQuestionTests(unittest.TestCase):
    def setUp(self):
        engine = types.ModuleType("app.engine")
        engine.search_db = Mock()
        engine.generate_answer = Mock(return_value="A sourced educational answer.")
        self.engine = engine
        sys.modules.pop("app.main", None)
        with patch.dict(sys.modules, {"app.engine": engine}):
            self.main = importlib.import_module("app.main")

    def tearDown(self):
        sys.modules.pop("app.main", None)

    def test_no_source_abstains_without_calling_model(self):
        self.engine.search_db.return_value = None

        response = self.main.ask_question("What is the treatment?")

        self.assertIn("cannot verify an answer", response["answer"])
        self.assertIsNone(response["source"])
        self.assertIsNone(response["retrieval_score"])
        self.engine.generate_answer.assert_not_called()

    def test_source_is_used_and_score_is_a_retrieval_indicator(self):
        self.engine.search_db.return_value = {
            "text": "A MedQuAD source answer.",
            "rerank_score": 0.0,
        }

        response = self.main.ask_question("A question")

        self.assertEqual(response["answer"], "A sourced educational answer.")
        self.assertEqual(response["source"], "A MedQuAD source answer.")
        self.assertEqual(response["retrieval_score"], 0.5)
        self.engine.generate_answer.assert_called_once_with(
            "A question", "A MedQuAD source answer."
        )


if __name__ == "__main__":
    unittest.main()
