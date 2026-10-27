"""Transactional inference tracking, separate from the refreshable instance cache."""
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import sys

INSTANCE_CACHE = Path.home() / ".cache/swebench/instances.db"
TASK_TABLES = {"<15 min fix": "tasks_short", "15 min - 1 hour": "tasks_medium"}
RUN_COLUMNS = {
    "run_id": "TEXT PRIMARY KEY", "instance_id": "TEXT NOT NULL",
    "trajectory_dir": "TEXT", "outcome": "TEXT", "routing": "INTEGER",
    "duration_seconds": "REAL", "model_patch_bytes": "INTEGER", "model": "TEXT",
    "session_file": "TEXT", "container_mem_peak_mb": "REAL",
    "container_mem_final_mb": "REAL", "host_pi_mem_peak_mb": "REAL",
    "resource_sample_errors": "INTEGER", "main_context_tokens": "INTEGER",
    "total_context_tokens": "INTEGER", "token_input": "INTEGER",
    "token_output": "INTEGER", "token_cache_read": "INTEGER",
    "token_cache_write": "INTEGER", "token_total": "INTEGER",
    "router_child_count": "INTEGER", "context_accounting_complete": "INTEGER",
    "missing_context_records": "INTEGER", "evaluation": "TEXT",
}
TOKEN_COLUMNS = dict(token_input="input", token_output="output", token_cache_read="cacheRead",
                     token_cache_write="cacheWrite", token_total="totalTokens")


def cache_difficulties(cache_path):
    """Never create or modify the source cache, including on a missing-path lookup."""
    path = Path(cache_path).expanduser().resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source:
        return dict(source.execute("SELECT instance_id, difficulty FROM instances"))


def _insert_run(connection, record, difficulties):
    run_id = Path(record["trajectory_dir"]).name.rsplit("__", 1)[-1]
    if not re.fullmatch(r"[0-9a-f]{8}", run_id):
        raise ValueError(f"Invalid trajectory run id: {run_id!r}")
    values = {key: record.get(key) for key in RUN_COLUMNS}
    # Early bundle-validation failures have no known instance, but are still runs.
    values.update(run_id=run_id, instance_id=record.get("instance_id") or "unknown", evaluation=None)
    usage = record.get("token_usage") or {}
    values.update({column: usage.get(key) for column, key in TOKEN_COLUMNS.items()})
    inserted = connection.execute(
        f"INSERT OR IGNORE INTO runs ({', '.join(values)}) VALUES ({', '.join('?' for _ in values)})",
        tuple(values.values()))
    if not inserted.rowcount:
        return  # Replays must not overwrite an external evaluation or reorder task ids.
    table = TASK_TABLES.get(difficulties.get(values["instance_id"]))
    if table:
        row = connection.execute(f"SELECT runs FROM {table} WHERE instance_id = ?",
                                 (values["instance_id"],)).fetchone()
        if row:
            ids = json.loads(row[0])
            if run_id not in ids:
                ids.append(run_id)
                connection.execute(f"UPDATE {table} SET runs = ? WHERE instance_id = ?",
                                   (json.dumps(ids), values["instance_id"]))


def initialize_runs_db(db_path, runs_log, cache_path=None):
    """Seed and backfill once, atomically; user_version marks successful initialization."""
    cache_path = INSTANCE_CACHE if cache_path is None else cache_path
    with closing(sqlite3.connect(db_path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        if connection.execute("PRAGMA user_version").fetchone()[0] >= 1:
            return
        if not Path(cache_path).expanduser().is_file():
            raise FileNotFoundError(f"Cannot seed runs database: instance cache missing: {cache_path}")
        difficulties = cache_difficulties(cache_path)
        for difficulty, table in TASK_TABLES.items():
            connection.execute(f"CREATE TABLE IF NOT EXISTS {table} (instance_id TEXT PRIMARY KEY, "
                               "difficulty TEXT NOT NULL, runs TEXT NOT NULL DEFAULT '[]')")
            connection.executemany(f"INSERT OR IGNORE INTO {table} (instance_id, difficulty) VALUES (?, ?)",
                                   [(key, value) for key, value in difficulties.items() if value == difficulty])
        connection.execute("CREATE TABLE IF NOT EXISTS runs (" + ", ".join(
            f"{key} {kind}" for key, kind in RUN_COLUMNS.items()) + ")")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_instance_id ON runs(instance_id)")
        log = Path(runs_log)
        if log.exists():
            for line in log.read_text(encoding="utf-8").splitlines()[-3:]:
                _insert_run(connection, json.loads(line), difficulties)
        connection.execute("PRAGMA user_version = 1")


def record_evaluation(db_path, run_id, verdict):
    """Store only 'success'/'fail' atomically; overwrite but never create a run."""
    if verdict not in ('success', 'fail'):
        raise ValueError("Evaluation verdict must be 'success' or 'fail'")
    path = Path(db_path).expanduser().resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        updated = connection.execute("UPDATE runs SET evaluation = ? WHERE run_id = ?",
                                     (verdict, run_id))
        if not updated.rowcount:
            raise ValueError(f"Run {run_id} is absent from runs database")


def migrate_evaluations(db_path):
    """Convert legacy JSON verdicts atomically; preserve NULLs and warn on invalid data."""
    path = Path(db_path).expanduser().resolve()
    migrated = 0
    with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        rows = connection.execute(
            "SELECT run_id, evaluation FROM runs WHERE evaluation IS NOT NULL "
            "AND evaluation NOT IN ('success', 'fail')").fetchall()
        for run_id, value in rows:
            try:
                legacy = json.loads(value)
                if not isinstance(legacy, dict) or 'resolved' not in legacy:
                    raise ValueError("missing resolved")
            except (ValueError, TypeError) as exc:
                print(f"Warning: run {run_id}: leaving evaluation untouched: {exc}", file=sys.stderr)
                continue
            verdict = 'success' if legacy['resolved'] is True else 'fail'
            connection.execute("UPDATE runs SET evaluation = ? WHERE run_id = ?", (verdict, run_id))
            migrated += 1
    return migrated


def append_run(db_path, record, cache_path=None):
    """Insert a run and update its seeded task list in a single serialized transaction."""
    cache_path = INSTANCE_CACHE if cache_path is None else cache_path
    try:
        difficulties = cache_difficulties(cache_path)
    except sqlite3.OperationalError:
        if Path(cache_path).expanduser().exists():
            raise
        difficulties = {}  # A removed cache must not prevent recording later runs.
    with closing(sqlite3.connect(db_path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        _insert_run(connection, record, difficulties)
