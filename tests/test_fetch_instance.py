"""Offline cache regression tests: python3 -m unittest test_fetch_instance."""
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import fetch_instance as fetch


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "instances.db"
        self.patch = patch.object(fetch, "DB_PATH", self.path)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def build(self, duplicate=False):
        import pyarrow as pa
        import pyarrow.parquet as pq
        row = {name: "text\nUnicode: λ" for name in (
            "base_commit", "created_at", "difficulty", "environment_setup_commit",
            "eval_type", "log_parser", "repo", "version", "patch", "test_patch",
            "eval_script", "problem_statement", "hints_text",
        )}
        row.update(image="example/image:latest", FAIL_TO_PASS=["test_one"], PASS_TO_PASS=[])
        rows = [dict(row, instance_id=f"example__repo-{i}") for i in range(500)]
        if duplicate:
            rows[-1] = rows[0]
        buffer = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(rows), buffer)
        manifest = {"parquet_files": [{"config": "default", "split": "test", "url": "https://example/shard"}]}
        with patch.object(fetch, "download", side_effect=[json.dumps(manifest).encode(), buffer.getvalue()]):
            fetch.build_database(self.path)
        return rows

    def test_roundtrip_cached_lookup_and_unique_index(self):
        rows = self.build()
        with patch.object(fetch, "download", side_effect=AssertionError("Unexpected network")):
            self.assertEqual(fetch.fetch_instance_row("example__repo-0"), rows[0])
            with self.assertRaisesRegex(ValueError, "found 0 rows"):
                fetch.fetch_instance_row("example__missing")
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM instances").fetchone()[0], 500)
            self.assertTrue(all(r[2] == "TEXT" for r in connection.execute("PRAGMA table_info(instances)")))
            self.assertEqual(connection.execute("PRAGMA index_list(instances)").fetchone()[2], 1)

    def test_duplicate_refresh_preserves_cache(self):
        rows = self.build()
        with self.assertRaises(sqlite3.IntegrityError):
            self.build(duplicate=True)
        self.assertEqual(fetch.fetch_instance_row("example__repo-499"), rows[-1])
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_validation_still_runs(self):
        self.build()
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE instances SET image = '-invalid'")
        with self.assertRaisesRegex(ValueError, "Invalid image"):
            fetch.fetch_instance_row("example__repo-0")

    def test_refresh_flag(self):
        self.build()
        with patch.object(fetch, "build_database") as build:
            fetch.fetch_instance_row("example__repo-0", refresh=True)
            build.assert_called_once_with(self.path)

    def test_missing_pyarrow_help(self):
        with patch.dict("sys.modules", {"pyarrow": None, "pyarrow.parquet": None}):
            with self.assertRaisesRegex(ValueError, "pip install pyarrow"):
                fetch.build_database(self.path)
        self.assertFalse(self.path.exists())

    def test_download_retries_are_bounded(self):
        with patch.object(fetch.urllib.request, "urlopen", side_effect=urllib.error.URLError("offline")) as request:
            with patch.object(fetch.time, "sleep") as sleep:
                with self.assertRaises(urllib.error.URLError):
                    fetch.download("https://example/shard")
        self.assertEqual(request.call_count, 3)
        self.assertEqual(sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
