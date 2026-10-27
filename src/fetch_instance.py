#!/usr/bin/env python3
"""Export one SWE-bench Verified instance for official evaluation.

Rows are cached in ~/.cache/swebench/instances.db; --refresh rebuilds the cache
from HuggingFace parquet shards. Building requires PyArrow (python3 -m pip
install pyarrow); cached lookups use only the Python standard library.
"""

import argparse
import io
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import tempfile
import urllib.error
import urllib.parse
import urllib.request


def validate_instance_id(instance_id: str) -> str:
    """Validate the instance identifier before looking it up."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*__[A-Za-z0-9_.-]+", instance_id):
        raise ValueError("Invalid instance_id format")
    return instance_id


DB_PATH = Path.home() / ".cache" / "swebench" / "instances.db"
# These two parquet list fields are JSON-encoded TEXT and decoded on lookup.
JSON_FIELDS = {"FAIL_TO_PASS", "PASS_TO_PASS"}
PARQUET_URL = (
    "https://datasets-server.huggingface.co/parquet?"
    + urllib.parse.urlencode({"dataset": "SWE-bench/SWE-bench_Verified"})
)


def download(url: str) -> bytes:
    """Retry only downloads, with three attempts and short backoff."""
    request = urllib.request.Request(url, headers={"User-Agent": "fetch-swebench-instance/3.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            if code < 500 or attempt == 2:
                raise
            reason = f"HTTP {code}"
        except (urllib.error.URLError, OSError) as exc:
            if attempt == 2:
                raise
            reason = str(exc)
        delay = attempt + 1
        print(f"Warning: {reason}; retry {attempt + 1}/2 in {delay}s", file=sys.stderr)
        time.sleep(delay)


def build_database(db_path: Path) -> None:
    """Build separately so failed refreshes never damage the existing cache."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError(
            "Building the instance cache requires PyArrow. Install it with "
            "'python3 -m pip install pyarrow' (use a virtual environment if needed)."
        ) from exc

    data = json.loads(download(PARQUET_URL))
    if not isinstance(data, dict) or data.get("pending") or data.get("failed") or data.get("partial"):
        raise ValueError("Parquet manifest is invalid or incomplete")
    files = data.get("parquet_files")
    if not isinstance(files, list):
        raise ValueError("Parquet manifest is missing parquet_files")
    urls = [
        item["url"] for item in files
        if isinstance(item, dict) and item.get("config") == "default"
        and item.get("split") == "test" and isinstance(item.get("url"), str)
    ]
    if not urls:
        raise ValueError("No parquet shards found for default/test")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="instances-", dir=db_path.parent) as temporary:
        staging = Path(temporary) / "instances.db"
        connection = sqlite3.connect(staging)
        try:
            columns = None
            count = 0
            for url in urls:
                table = pq.read_table(io.BytesIO(download(url)))
                if columns is None:
                    columns = table.column_names
                    if len(columns) != 17 or len(set(columns)) != 17 or "instance_id" not in columns:
                        raise ValueError("Expected 17 distinct fields including instance_id")
                    quoted = ['"' + name.replace('"', '""') + '"' for name in columns]
                    definitions = [
                        name + " TEXT" + (" NOT NULL" if field == "instance_id" else "")
                        for name, field in zip(quoted, columns)
                    ]
                    connection.execute("CREATE TABLE instances (" + ", ".join(definitions) + ")")
                    connection.execute("CREATE UNIQUE INDEX instances_instance_id ON instances(instance_id)")
                elif table.column_names != columns:
                    raise ValueError("Parquet shards have inconsistent fields")
                rows = table.to_pylist()
                for row in rows:
                    for name, value in row.items():
                        if name in JSON_FIELDS:
                            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                                raise ValueError(f"Expected a list of strings in {name}")
                            row[name] = json.dumps(value, ensure_ascii=False)
                        elif value is not None and not isinstance(value, str):
                            raise ValueError(f"Expected string or null in {name}")
                connection.executemany(
                    "INSERT INTO instances VALUES (" + ", ".join("?" for _ in columns) + ")",
                    ([row[name] for name in columns] for row in rows),
                )
                count += len(rows)
            if count != 500:
                raise ValueError(f"Expected 500 Verified instances, got {count}")
            connection.commit()
        finally:
            connection.close()
        staging.replace(db_path)


def fetch_instance_row(instance_id: str, refresh: bool = False) -> dict:
    """Look up exactly one cached row, then apply local validation."""
    validate_instance_id(instance_id)
    if refresh or not DB_PATH.exists():
        build_database(DB_PATH)
    connection = sqlite3.connect(DB_PATH.as_uri() + "?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM instances WHERE instance_id = ?", (instance_id,)
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != 1:
        raise ValueError(f"Could not find a unique instance {instance_id}: found {len(rows)} rows")
    row = dict(rows[0])
    for name in JSON_FIELDS:
        row[name] = json.loads(row[name])
    if row.get("instance_id") != instance_id:
        raise ValueError("Database returned a mismatched instance_id")
    for field in ("image", "eval_script", "patch", "test_patch"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"Data is missing a nonempty string field: {field}")
    if any(c.isspace() for c in row["image"]) or row["image"].startswith("-"):
        raise ValueError("Invalid image reference format in the data")
    return row


def write_bundle(row: dict, out_dir: Path) -> list[Path]:
    """Write generated files, preserving unrelated files in an existing directory.

    The caller is responsible for enforcing its overwrite policy.
    Patch and script bytes are exactly their original UTF-8 field contents.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contents = {
        "instance.json": json.dumps([row], ensure_ascii=False, indent=2) + "\n",
        "eval.sh": row["eval_script"],
        "image.txt": row["image"] + "\n",
        "gold.patch": row["patch"],
        "test_patch.diff": row["test_patch"],
    }
    paths = []
    for name, content in contents.items():
        path = out_dir / name
        path.write_bytes(content.encode("utf-8"))
        paths.append(path)
    (out_dir / "eval.sh").chmod(0o755)
    return paths


def pull_image(image: str) -> bool:
    """Pull the amd64 image; Docker absence or failure is nonfatal."""
    if shutil.which("docker") is None:
        print("Warning: Docker not found; downloading evaluation files only. Pull the image manually later.", file=sys.stderr)
        return False
    print(f"Pulling image: {image}", flush=True)
    try:
        subprocess.run(["docker", "pull", "--platform", "linux/amd64", image], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Warning: Image pull failed; pull it manually later: {exc}", file=sys.stderr)
        return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_id", help="Example: astropy__astropy-12907")
    parser.add_argument("output_dir", nargs="?", type=Path, help="Default: ./<instance_id>/")
    parser.add_argument("--force", action="store_true", help="Overwrite generated files and preserve other files")
    parser.add_argument("--no-pull", action="store_true", help="Skip pulling the Docker image")
    parser.add_argument("--refresh", action="store_true", help="Rebuild the SQLite cache from parquet")
    args = parser.parse_args(argv)
    try:
        validate_instance_id(args.instance_id)
        out_dir = args.output_dir if args.output_dir is not None else Path(args.instance_id)
        if out_dir.exists() or out_dir.is_symlink():
            if not args.force:
                raise ValueError(f"Output path already exists: {out_dir}; use --force to overwrite generated files")
            if not out_dir.is_dir():
                raise ValueError(f"Output path is not a directory: {out_dir}")
            print(f"Note: Generated files in {out_dir} will be overwritten; other files will be preserved.", file=sys.stderr)
        print(f"Fetching instance: {args.instance_id}", flush=True)
        row = fetch_instance_row(args.instance_id, refresh=args.refresh)
        write_bundle(row, out_dir)
        if not args.no_pull:
            pull_image(row["image"])
    except (OSError, ValueError, urllib.error.URLError, sqlite3.Error) as exc:
        print(f"Error: Failed to fetch or write the instance: {exc}; the output directory may contain incomplete files.", file=sys.stderr)
        return 1
    print(f"Done: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
