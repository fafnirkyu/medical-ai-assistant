"""Offline ingestion tests using a regular SQLite table in place of vec0."""

import importlib
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


class FakeVecConnection(sqlite3.Connection):
    def execute(self, sql, parameters=()):
        if "CREATE VIRTUAL TABLE vec_medquad USING vec0" in sql:
            sql = "CREATE TABLE vec_medquad (rowid INTEGER PRIMARY KEY, embedding BLOB)"
        return super().execute(sql, parameters)


class FakeVector(list):
    def tolist(self):
        return list(self)


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db_path = Path(self.directory.name) / "medical_data.db"

        with sqlite3.connect(self.db_path, factory=FakeVecConnection) as db:
            db.execute(
                "CREATE TABLE medquad (id INTEGER PRIMARY KEY, question TEXT, answer TEXT)"
            )
            db.execute(
                "CREATE TABLE vec_medquad (rowid INTEGER PRIMARY KEY, embedding BLOB)"
            )
            db.execute("INSERT INTO medquad VALUES (1, 'old question', 'old answer')")
            db.execute("INSERT INTO vec_medquad VALUES (1, X'00')")

        vec = types.ModuleType("sqlite_vec")
        vec.load = Mock()
        datasets = types.ModuleType("datasets")
        datasets.load_dataset = Mock(
            return_value=[
                {"question": "first", "answer": "answer one"},
                {"question": "second", "answer": "answer two"},
            ]
        )
        transformers = types.ModuleType("sentence_transformers")
        self.model = Mock()
        self.model.encode.return_value = FakeVector([0.0] * 384)
        transformers.SentenceTransformer = Mock(return_value=self.model)

        sys.modules.pop("app.ingest_data", None)
        with (
            patch.dict(
                sys.modules,
                {
                    "sqlite_vec": vec,
                    "datasets": datasets,
                    "sentence_transformers": transformers,
                },
            ),
            patch.dict(os.environ, {"DATABASE_PATH": str(self.db_path)}),
        ):
            self.ingest = importlib.import_module("app.ingest_data")

        self.original_connect = sqlite3.connect
        self.connect_patch = patch.object(
            self.ingest.sqlite3,
            "connect",
            side_effect=lambda path: self.original_connect(
                path, factory=FakeVecConnection
            ),
        )
        self.connect_patch.start()
        self.addCleanup(self.connect_patch.stop)
        self.addCleanup(lambda: sys.modules.pop("app.ingest_data", None))

    def read_answers(self):
        with self.original_connect(self.db_path, factory=FakeVecConnection) as db:
            return db.execute("SELECT answer FROM medquad ORDER BY id").fetchall()

    def test_success_rebuilds_matching_tables(self):
        self.ingest.setup_database()
        self.assertEqual(self.read_answers(), [("answer one",), ("answer two",)])
        with self.original_connect(self.db_path, factory=FakeVecConnection) as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM vec_medquad").fetchone()[0], 2
            )

    def test_embedding_failure_rolls_back_to_previous_database(self):
        self.model.encode.side_effect = [
            FakeVector([0.0] * 384),
            RuntimeError("embedding failed"),
        ]

        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            self.ingest.setup_database()

        self.assertEqual(self.read_answers(), [("old answer",)])
        with self.original_connect(self.db_path, factory=FakeVecConnection) as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM vec_medquad").fetchone()[0], 1
            )


if __name__ == "__main__":
    unittest.main()
