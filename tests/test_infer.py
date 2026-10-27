"""Offline runner tests plus opt-in Docker/pi-loader tool smoke test."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
from unittest.mock import patch

import pytest
import infer
import run_tracking


@pytest.fixture(autouse=True)
def isolated_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(infer, "RUNS_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(infer, "REPO_ROOT", tmp_path)
    cache = tmp_path / "instances.db"
    with sqlite3.connect(cache) as connection:
        connection.execute("CREATE TABLE instances (instance_id TEXT, difficulty TEXT)")
        connection.executemany("INSERT INTO instances VALUES (?, ?)", [
            ("short", "<15 min fix"), ("astropy__astropy-12907", "15 min - 1 hour"),
            ("long", "1-4 hours"), ("very-long", ">4 hours")])
    monkeypatch.setattr(run_tracking, "INSTANCE_CACHE", cache)
    # Unit lifecycle mocks must not start Docker sampling threads.
    if not os.environ.get("SWE_INFER_IT") and not os.environ.get("SWE_INFER_ROUTING_IT"):
        monkeypatch.setattr(infer.ResourceSampler, "loop", lambda self: None)
        monkeypatch.setattr(infer.ResourceSampler, "sample", lambda self: None)


def tracking_record(run_id, instance_id="astropy__astropy-12907", **fields):
    return dict(instance_id=instance_id, trajectory_dir=f"/trajectories/stamp__plain__{run_id}",
                token_usage=dict(input=10, output=20, cacheRead=30, cacheWrite=40, totalTokens=100),
                **fields)


def test_runs_schema_and_seed(tmp_path):
    db = tmp_path / "runs.db"
    cache = run_tracking.INSTANCE_CACHE
    original = cache.read_bytes()
    run_tracking.initialize_runs_db(db, infer.RUNS_LOG)
    with sqlite3.connect(db) as connection:
        assert set(row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")) == {"tasks_short", "tasks_medium", "runs"}
        assert list(connection.execute("SELECT * FROM tasks_short")) == [("short", "<15 min fix", "[]")]
        assert list(connection.execute("SELECT * FROM tasks_medium")) == [
            ("astropy__astropy-12907", "15 min - 1 hour", "[]")]
        assert {row[1] for row in connection.execute("PRAGMA table_info(runs)")} == set(run_tracking.RUN_COLUMNS)
        assert "idx_runs_instance_id" in {row[1] for row in connection.execute("PRAGMA index_list(runs)")}
    run_tracking.initialize_runs_db(db, infer.RUNS_LOG)
    assert cache.read_bytes() == original


@pytest.mark.parametrize("outcome,routing", [("ok", False), ("error", True), ("cancelled", True)])
def test_runs_append(tmp_path, outcome, routing):
    db = tmp_path / "runs.db"
    run_tracking.initialize_runs_db(db, infer.RUNS_LOG)
    record = tracking_record("12345678", outcome=outcome, routing=routing, evaluation="ignored")
    run_tracking.append_run(db, record)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE runs SET evaluation = 'external verdict'")
    run_tracking.append_run(db, record)
    run_tracking.append_run(db, tracking_record("23456789"))
    run_tracking.append_run(db, tracking_record("34567890", "short"))
    run_tracking.append_run(db, tracking_record("45678901", "unknown-instance"))
    run_tracking.append_run(db, tracking_record("56789012", "long"))
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 5
        assert json.loads(connection.execute("SELECT runs FROM tasks_medium").fetchone()[0]) == ["12345678", "23456789"]
        assert json.loads(connection.execute("SELECT runs FROM tasks_short").fetchone()[0]) == ["34567890"]
        assert connection.execute("SELECT outcome, routing, token_input, token_output, token_cache_read, "
                                  "token_cache_write, token_total, evaluation FROM runs WHERE run_id='12345678'").fetchone() == (
                                      outcome, int(routing), 10, 20, 30, 40, 100, "external verdict")
        assert connection.execute("SELECT count(*) FROM runs WHERE evaluation IS NULL").fetchone()[0] == 4


def test_runs_backfill_last_three_once(tmp_path):
    db = tmp_path / "runs.db"
    ids = ["00000001", "00000002", "00000003", "aa69feb4", "5560aeb1", "442bb25a"]
    infer.RUNS_LOG.write_text("\n".join(json.dumps(tracking_record(key)) for key in ids) + "\n")
    run_tracking.initialize_runs_db(db, infer.RUNS_LOG)
    infer.RUNS_LOG.write_text(json.dumps(tracking_record("ffffffff")) + "\n")
    run_tracking.initialize_runs_db(db, infer.RUNS_LOG)
    with sqlite3.connect(db) as connection:
        assert {row[0] for row in connection.execute("SELECT run_id FROM runs")} == set(ids[-3:])
        assert json.loads(connection.execute("SELECT runs FROM tasks_medium").fetchone()[0]) == ids[-3:]
        assert connection.execute("SELECT count(*) FROM runs WHERE evaluation IS NULL").fetchone()[0] == 3


def test_runs_missing_cache_seed_retry_and_append(tmp_path):
    db = tmp_path / "runs.db"
    missing = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError, match="Cannot seed runs database: instance cache missing"):
        run_tracking.initialize_runs_db(db, infer.RUNS_LOG, missing)
    assert not missing.exists()
    run_tracking.initialize_runs_db(db, infer.RUNS_LOG)
    run_tracking.initialize_runs_db(db, infer.RUNS_LOG, missing)
    run_tracking.append_run(db, tracking_record("12345678"), missing)
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    assert not missing.exists()


def test_trajectory_naming(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    import uuid
    monkeypatch.chdir(tmp_path.parent)
    directory = infer.trajectory_directory("records", "a/b", True,
        datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc), uuid.UUID(int=0))
    assert directory == tmp_path / "records/20260102T030405.000000Z__a_b__routing__00000000"
    assert "__plain__" in infer.trajectory_directory(tmp_path, "id", False).name
    assert infer.trajectory_directory(tmp_path, "id", False) != infer.trajectory_directory(tmp_path, "id", False)
    assert not directory.exists()


def test_persist_trajectory(tmp_path):
    source = tmp_path / "source.jsonl"
    raw = b'{"type":"session","id":"main"}\r\n{"partial":\xff'
    source.write_bytes(raw)
    router = tmp_path / "router"
    router.mkdir()
    nodes = [dict(agent_run_id="root", session_id="main", entries={"child": {}}),
             dict(agent_run_id="child", entries={"grandchild": {}}),
             dict(agent_run_id="grandchild", entries={"root": {}}, detail={"events": []}),
             dict(agent_run_id="unrelated", session_id="other", entries={})]
    for node in nodes:
        infer.write_json(router / (node["agent_run_id"] + ".json"), node)
    directory = tmp_path / "record"
    meta = dict(instance_id="id", head_commit="abc", git_status_at_start="")
    infer.initialize_trajectory(directory, "exact prompt\n\né", meta)
    metrics = {"routing": True, "outcome": "cancelled"}
    result = infer.persist_trajectory(directory, source, metrics, router)
    assert "trajectory_dir" not in metrics
    assert result == dict(metrics, trajectory_dir=str(directory))
    assert json.loads((directory / "run.json").read_text()) == result
    assert (directory / "trajectory.jsonl").read_bytes() == raw
    assert (directory / "prompt.txt").read_bytes() == "exact prompt\n\né".encode()
    assert json.loads((directory / "meta.json").read_text()) == meta
    assert sorted(p.name for p in (directory / "children").iterdir()) == ["child.json", "grandchild.json", "root.json"]
    for name in ("root", "child", "grandchild"):
        assert (directory / "children" / (name + ".json")).read_bytes() == (router / (name + ".json")).read_bytes()
    infer.persist_trajectory(directory, directory / "trajectory.jsonl", metrics, router)
    assert (directory / "trajectory.jsonl").read_bytes() == raw


def test_persist_missing_session(tmp_path):
    directory = tmp_path / "record"
    infer.initialize_trajectory(directory, "", {})
    infer.persist_trajectory(directory, tmp_path / "missing", {"routing": False})
    assert (directory / "trajectory.jsonl").read_bytes() == b""
    assert (directory / "children/README.md").read_text() == "no router children spawned for this run\n"


def test_memory_metrics():
    assert infer.memory_mb("1.5GiB / 7.746GiB") == 1536
    assert infer.memory_mb("1048576B / 2GiB") == 1
    assert infer.memory_mb("1MB / 2GB") == 1000000 / 1024**2
    with pytest.raises(ValueError):
        infer.memory_mb("unavailable")
    assert infer.tree_rss_mb("10 1 10 1024\n11 10 10 2048\n12 11 12 512\n99 1 99 9000", 10) == 3.5


def test_usage_context_is_not_cumulative():
    events = [
        {"type": "message", "message": {"role": "assistant", "usage":
         {"input": 100, "output": 10, "cacheRead": 20, "cacheWrite": 5}}},
        {"type": "compaction", "usage": {"input": 1000, "output": 100}},
        {"type": "message", "message": {"role": "assistant", "usage":
         {"input": 30, "output": 2}}},
        {"type": "message", "message": {"role": "assistant", "usage": {"input": 0, "output": 0}}},
        {"type": "message", "message": {"role": "toolResult", "usage": {"input": 999}}},
    ]
    context, usage = infer.usage_metrics(events)
    assert context == 32
    assert usage["totalTokens"] == 1267
    assert infer.usage_metrics([])[0] is None


def test_context_accounting(tmp_path):
    def message(n):
        return {"type": "message_end", "message": {"role": "assistant", "usage":
                {"input": n, "output": 5, "cacheRead": 3, "cacheWrite": 2}}}
    session = tmp_path / "session.jsonl"
    rows = [{"type": "session", "id": "main"}, message(10), message(20)]
    session.write_text("\n".join(map(json.dumps, rows)) + '\n{"partial":')
    root = {"agent_run_id": "root", "session_id": "main", "entries": {"child": {}}}
    child = {"agent_run_id": "child", "entries": {"grandchild": {}}, "detail": {"events": [message(30)]}}
    grandchild = {"agent_run_id": "grandchild", "detail": {"events": [message(40)]}}
    for node in (root, child, grandchild):
        (tmp_path / (node["agent_run_id"] + ".json")).write_text(json.dumps(node))
    result = infer.context_metrics(session, True, tmp_path)
    assert result["main_context_tokens"] == 30
    assert result["total_context_tokens"] == 120
    assert result["token_usage"]["totalTokens"] == 140
    assert result["router_child_count"] == 2
    assert result["context_accounting_complete"]
    (tmp_path / "grandchild.json").unlink()
    result = infer.context_metrics(session, True, tmp_path)
    assert result["missing_context_records"] == 1
    assert not result["context_accounting_complete"]
    assert infer.context_metrics(None, False)["main_context_tokens"] is None
    plain = infer.context_metrics(session, False)
    assert plain["total_context_tokens"] == plain["main_context_tokens"]


def bundle(tmp_path):
    (tmp_path / "image.txt").write_text("python:3.11-slim\n")
    (tmp_path / "instance.json").write_text(json.dumps([{
        "instance_id": "example__issue-1", "problem_statement": "Fix the example issue",
        "patch": "SECRET_GOLD", "test_patch": "SECRET_TEST",
    }]))
    return tmp_path


def test_commands():
    assert infer.docker_run("image", "name", "/repo") == [
        "docker", "run", "-d", "--init", "-it", "--name", "name", "-w", "/repo", "image", "sleep", "infinity"]
    assert infer.docker_exec("cid", "git", "status") == ["docker", "exec", "cid", "git", "status"]
    assert "cd '/a b' && git add -N ." in infer.capture_command("cid", "/a b")[-1]
    plain = infer.pi_command("test", "hello")
    routed = infer.pi_command("test", "hello", routing=True)
    assert "--no-extensions" in plain
    assert plain == [
        "pi", "--print", "--no-session", "--no-extensions", "--no-mcp",
        "--no-skills", "--no-prompt-templates", "--no-context-files", "--no-approve",
        "--tools", "bash,read,edit,write", "--extension", str(infer.EXTENSION),
        "--model", "test", "--", "hello"]
    expected = [arg for arg in plain if arg != "--no-extensions"]
    expected[expected.index("--tools") + 1] += (
        ",darknavstar-get-model-pool,darknavstar-spawn-model-pool"
        ",darknavstar-get-agent-run-records")
    assert routed == expected
    assert routed[routed.index("--extension") + 1] == str(infer.EXTENSION)


def test_routing_settings(tmp_path, monkeypatch):
    source = tmp_path / "user"
    source.mkdir()
    (source / "auth.json").write_text("{}")
    original = {"packages": ["../../router", {"source": "npm:example", "skills": []}],
                "extensions": ["./extra.ts", "-builtin:mcp"], "defaultTools": ["grep"]}
    (source / "settings.json").write_text(json.dumps(original))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(source))
    target = tmp_path / "overlay"
    infer.routing_settings(target)
    settings = json.loads((target / "settings.json").read_text())
    assert settings["packages"][0] == str(source / "../../router")
    assert settings["packages"][1] == original["packages"][1]
    assert str(infer.EXTENSION) in settings["extensions"]
    assert "+" + str(infer.EXTENSION) in settings["extensions"]
    assert settings["defaultTools"] == ["bash", "read", "edit", "write"]
    assert (target / "auth.json").resolve() == source / "auth.json"
    assert json.loads((source / "settings.json").read_text()) == original


def test_bundle_and_prediction(tmp_path):
    assert infer.read_bundle(bundle(tmp_path)) == ("python:3.11-slim", "example__issue-1")
    line = infer.prediction_line("id", "model", "diff\n+é\n")
    assert len(line.splitlines()) == 1
    assert json.loads(line)["model_patch"] == "diff\n+é\n"
    (tmp_path / "image.txt").write_text("--bad\n")
    with pytest.raises(ValueError):
        infer.read_bundle(tmp_path)


@pytest.mark.parametrize("routing", [False, True])
def test_dry_run(tmp_path, capsys, routing):
    with patch("infer.subprocess.run", side_effect=AssertionError), patch("infer.subprocess.Popen", side_effect=AssertionError):
        assert infer.main([str(bundle(tmp_path)), "--model", "test", "--dry-run"] +
                          (["--routing"] if routing else [])) == 0
    output = capsys.readouterr().out
    assert "Fix the example issue" in output
    assert "SECRET_GOLD" not in output
    assert "SECRET_TEST" not in output
    assert "# trajectory dir: " in output
    assert not (tmp_path / "trajectories").exists()
    assert not (tmp_path / "runs.db").exists()
    assert "docker run -d --init -it" in output
    assert "git add -N ." in output
    assert ("--no-extensions" in output) is not routing
    assert ("PI_CODING_AGENT_DIR=" in output) is routing
    assert not (tmp_path.parent / "predictions.jsonl").exists()


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("routing", [False, True])
def test_capture_before_removal(tmp_path, failure, routing, capsys):
    calls = []
    def run(command):
        calls.append(command)
        if command[1] == "run":
            return "cid\n"
        if "status" in command:
            return "dirty" if failure else ""
        if command == infer.capture_command("cid", "/testbed"):
            return "diff\n+é\r\n"
        return "head\n"
    with patch("infer.run", side_effect=run), patch("infer.subprocess.Popen") as popen:
        popen.return_value.wait.return_value = 0
        popen.return_value.poll.return_value = 0
        out = tmp_path / "predictions.jsonl"
        db = tmp_path / ("custom.db" if routing else "runs.db")
        assert infer.main([str(bundle(tmp_path)), "--model", "test", "--predictions-out", str(out)] +
                          (["--routing", "--runs-db", str(db)] if routing else [])) == int(failure)
        if not failure:
            env = popen.call_args.kwargs["env"]
            assert env["SWE_INFER_CONTAINER"] == "cid"
            if routing:
                assert not Path(env["PI_CODING_AGENT_DIR"]).exists()
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["routing"] is routing
    directory = Path(metrics["trajectory_dir"])
    assert json.loads((directory / "run.json").read_text()) == metrics
    assert (directory / "trajectory.jsonl").exists()
    meta = json.loads((directory / "meta.json").read_text())
    assert meta["container_id"] == "cid"
    assert meta["git_status_at_start"] == ("dirty" if failure else "")
    assert json.loads((tmp_path / "runs.jsonl").read_text()) == metrics
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT run_id, outcome, routing, evaluation FROM runs").fetchall() == [
            (directory.name.rsplit("__", 1)[-1], metrics["outcome"], int(routing), None)]
    assert metrics["main_context_tokens"] is None
    assert metrics["container_mem_peak_mb"] is None
    assert calls[-2] == infer.capture_command("cid", "/testbed")
    assert calls[-1] == ["docker", "rm", "-f", "cid"]
    assert json.loads(out.read_text())["model_patch"] == "diff\n+é\r\n"
    assert (directory / "patch.diff").read_bytes() == "diff\n+é\r\n".encode("utf-8")


@pytest.mark.parametrize("spawn", [None, False, True])
def test_router_spawn_receipt_outcome(tmp_path, capsys, spawn):
    router = tmp_path / ".pi/model-router-trajectory"
    router.mkdir(parents=True)
    events = [{"type": "session", "id": "main"},
              {"type": "message", "message": {"role": "assistant",
               "usage": {"input": 10, "output": 20}}}]
    if spawn is not None:
        events.append({"type": "message", "message": {
            "role": "toolResult", "toolName": "darknavstar-spawn-model-pool",
            "content": [{"type": "text", "text": json.dumps({"receipt": {
                "fallback_to_main_loop": spawn}})}]}})
    (router / "root.json").write_text(json.dumps({"agent_run_id": "root",
        "session_id": "main", "entries": {"child": {}} if spawn is not None else {}}))
    if spawn is False:
        (router / "child.json").write_text(json.dumps({"agent_run_id": "child",
            "detail": {"events": events[1:2]}}))

    def fake_pi(command, **kwargs):
        Path(command[command.index("--session") + 1]).write_text(
            "\n".join(json.dumps(e) for e in events))
        from unittest.mock import Mock
        return Mock(pid=123, wait=lambda **kw: 0, poll=lambda: 0)

    def fake_docker(command):
        if command[1] == "run":
            return "cid"
        if command == infer.capture_command("cid", "/testbed"):
            return "diff\n+patch\n"
        return ""

    with patch("infer.Path.home", return_value=tmp_path), \
            patch("infer.routing_settings"), \
            patch("infer.run", side_effect=fake_docker), \
            patch("infer.subprocess.Popen", side_effect=fake_pi):
        result = infer.main([str(bundle(tmp_path)), "--model", "test", "--routing",
                             "--predictions-out", str(tmp_path / "predictions.jsonl")])
    record = json.loads(capsys.readouterr().out)
    assert result == (1 if spawn is True else 0)
    assert record["outcome"] == ("fail" if spawn is True else "ok")
    assert record["router_child_count"] == (1 if spawn is False else 0)
    assert record["missing_context_records"] == 0
    assert record["context_accounting_complete"] is True
    assert record["total_context_tokens"] == (60 if spawn is False else 30)
    directory = Path(record["trajectory_dir"])
    assert (directory / "patch.diff").read_text() == "diff\n+patch\n"
    assert json.loads((directory / "run.json").read_text()) == record
    assert json.loads((tmp_path / "runs.jsonl").read_text()) == record
    with sqlite3.connect(tmp_path / "runs.db") as db:
        assert db.execute("SELECT outcome, router_child_count FROM runs").fetchone() == (
            record["outcome"], record["router_child_count"])


@pytest.mark.parametrize("text,expected", [
    ('prefix {"fallback_to_main_loop": true} truncated', True),
    ('{"receipt":{"fallback_to_main_loop":false}}', False),
    ('{"fallback_to_main_loop":"true"}', False),
])
def test_router_spawn_fallback_text(text, expected):
    event = {"message": {"role": "toolResult", "toolName": "darknavstar-spawn-model-pool",
                         "content": [{"type": "text", "text": text}]}}
    assert infer.router_spawn_fallback([event]) is expected
    event["message"]["toolName"] = "bash"
    assert not infer.router_spawn_fallback([event])


@pytest.mark.parametrize("present", [True, False])
def test_local_image_required(tmp_path, capsys, present):
    calls = []

    def fake_docker(command):
        calls.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            if not present:
                raise subprocess.CalledProcessError(1, command)
            return "[]\n"
        if command[1] == "run":
            return "cid\n"
        return ""

    with patch("infer.run", side_effect=fake_docker), patch("infer.subprocess.Popen") as popen:
        popen.return_value.wait.return_value = 0
        popen.return_value.poll.return_value = 0
        result = infer.main([str(bundle(tmp_path)), "--model", "test",
                             "--predictions-out", str(tmp_path / "predictions.jsonl")])
    captured = capsys.readouterr()
    assert calls[0] == ["docker", "image", "inspect", "python:3.11-slim"]
    assert result == (0 if present else 1)
    assert json.loads(captured.out)["outcome"] == ("ok" if present else "error")
    if present:
        assert calls[1][1] == "run"
        popen.assert_called_once()
    else:
        assert calls == [["docker", "image", "inspect", "python:3.11-slim"]]
        popen.assert_not_called()
        assert "python:3.11-slim" in captured.err
        assert "Pull it first" in captured.err
        assert "fetch_instance.py example__issue-1" in captured.err
        assert "docker pull --platform linux/amd64 python:3.11-slim" in captured.err


def test_default_prompt_requires_issue_and_allows_override(tmp_path, capsys):
    directory = bundle(tmp_path)
    (directory / "instance.json").write_text('[{"instance_id":"example__issue-1"}]')
    assert infer.main([str(directory), "--model", "test", "--dry-run"]) == 1
    assert "no problem_statement" in capsys.readouterr().err
    assert infer.main([str(directory), "--model", "test", "--dry-run", "--prompt", "Explicit issue"]) == 0
    assert "Explicit issue" in capsys.readouterr().out


def test_cancel_and_keep(tmp_path):
    calls = []
    def run(command):
        calls.append(command)
        return "cid" if command[1] == "run" else ""
    with patch("infer.run", side_effect=run), patch("infer.subprocess.Popen") as popen:
        popen.return_value.wait.side_effect = KeyboardInterrupt
        popen.return_value.poll.return_value = 1
        assert infer.main([str(bundle(tmp_path)), "--model", "test", "--keep-container",
                           "--predictions-out", str(tmp_path / "predictions.jsonl")]) == 130
    assert calls[-1] == infer.capture_command("cid", "/testbed")
    assert not any(command[1] == "rm" for command in calls)
    record = json.loads((tmp_path / "runs.jsonl").read_text())
    assert (Path(record["trajectory_dir"]) / "patch.diff").read_bytes() == b""


@pytest.mark.skipif(os.environ.get("SWE_INFER_ROUTING_IT") != "1", reason="Set SWE_INFER_ROUTING_IT=1 for child CLI smoke")
def test_routing_child_cli(tmp_path):
    """Replicate dispatch.ts's exact child CLI with a deterministic local model.

    No extra --extension docker-tools flag: discovery MUST use the inherited overlay.
    This checks real CLI startup, router child validation, and model-issued tools.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    import uuid

    router = Path(os.environ.get("SWE_ROUTER_PATH", "/Users/cc/workplace/darknavstar-model-router"))
    image_bundle = Path.home() / ".cache/swebench/astropy__astropy-12907"
    image, _ = infer.read_bundle(image_bundle)
    results = []
    commands = [
        ("write", {"path": "routing-probe.txt", "content": "container marker"}),
        ("edit", {"path": "routing-probe.txt", "edits": [{"oldText": "marker", "newText": "edited"}]}),
        ("read", {"path": "routing-probe.txt"}),
        ("bash", {"command": "pwd; uname -s; cat /testbed/routing-probe.txt"}),
    ]

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            tool_results = [m for m in request["messages"] if m["role"] == "tool"]
            results[:] = tool_results
            index = len(tool_results)
            if index < len(commands):
                name, arguments = commands[index]
                delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": f"call_{index}",
                    "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}
                finish = "tool_calls"
            else:
                delta, finish = {"role": "assistant", "content": "Verified container tools."}, "stop"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for body, reason in [(delta, None), ({}, finish)]:
                chunk = {"id": "probe", "object": "chat.completion.chunk", "created": 1,
                         "model": "probe", "choices": [{"index": 0, "delta": body, "finish_reason": reason}]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cid = infer.run(infer.docker_run(image, "swebench-routing-test-" + uuid.uuid4().hex, "/testbed")).strip()
    try:
        agent_dir = tmp_path / "agent"
        infer.routing_settings(agent_dir)
        models = agent_dir / "models.json"
        if models.is_symlink():
            models.unlink()
        models.write_text(json.dumps({"providers": {"routing-probe": {
            "baseUrl": f"http://127.0.0.1:{server.server_port}/v1", "api": "openai-completions",
            "apiKey": "test", "models": [{"id": "probe", "contextWindow": 32000, "maxTokens": 1000}]}}}))
        child = dict(runId=str(uuid.uuid4()), parentRunId=str(uuid.uuid4()), projectDir=str(tmp_path),
                     depth=1, modelConfig={"model_alias": "routing-probe/probe", "censored": False},
                     route={"category": "coding", "role": "read-write", "task_complexity": "simple"},
                     allowedTools=["bash", "read", "edit", "write"])
        child_file = tmp_path / "child.json"
        child_file.write_text(json.dumps(child))
        env = dict(os.environ, PI_CODING_AGENT_DIR=str(agent_dir), SWE_INFER_CONTAINER=cid,
                   SWE_INFER_WORKDIR="/testbed", PI_MODEL_ROUTER_CHILD_CONFIG=str(child_file),
                   PI_MODEL_ROUTER_RUN_ID=child["runId"], PI_MODEL_ROUTER_PARENT_RUN_ID=child["parentRunId"],
                   PI_OFFLINE="1")
        command = ["pi", "--mode", "json", "-p", "--no-session", "--provider", "routing-probe",
                   "--model", "probe", "--thinking", "off", "-e", str(router / "darknavstar-model-router.ts"),
                   "--tools", "bash,read,edit,write", "--", "Task:\nVerify container tools"]
        completed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
        assert completed.returncode == 0, completed.stderr + completed.stdout
        assert '"type":"model_pool_agent_started"' in completed.stdout
        assert len(results) == 4, completed.stderr + completed.stdout
        assert "container edited" in json.dumps(results[2])
        assert "/testbed" in json.dumps(results[3]) and "Linux" in json.dumps(results[3])
        assert "container edited" in json.dumps(results[3])
        assert not (tmp_path / "routing-probe.txt").exists()
        print("PASS: exact router child CLI; inherited overlay; write/edit/read/bash inside Docker")
    finally:
        infer.run(["docker", "rm", "-f", cid])
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.skipif(os.environ.get("SWE_INFER_IT") != "1", reason="Set SWE_INFER_IT=1 for Docker smoke")
def test_docker_tools(tmp_path):
    if not shutil.which("docker") or subprocess.run(["docker", "info"], capture_output=True).returncode:
        pytest.skip("Docker unavailable")
    image = "python:3.11-slim"
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode:
        pytest.skip("Small local image unavailable; tests never pull")
    root = Path("/opt/homebrew/lib/node_modules/@earendil-works/pi-coding-agent")
    if not root.exists() or not shutil.which("node"):
        pytest.skip("Installed pi/node unavailable")
    cid = infer.run(infer.docker_run(image, "swebench-infer-test-" + str(os.getpid()), "/tmp")).strip()
    try:
        script = tmp_path / "smoke.mjs"
        script.write_text('''
import assert from 'node:assert/strict';
import {loadExtensions} from 'ROOT/dist/core/extensions/loader.js';
const loaded = await loadExtensions([process.env.EXTENSION], process.cwd());
assert.deepEqual(loaded.errors, []);
const tools = loaded.extensions[0].tools;
assert.deepEqual([...tools.keys()].sort(), ['bash','edit','read','write']);
const call = (name, params, signal) => tools.get(name).definition.execute('test', params, signal, undefined, {cwd:'/tmp'});
await call('write', {path:'container-only.txt', content:'container marker\\n'});
assert.match(JSON.stringify(await call('read', {path:'container-only.txt'})), /container marker/);
await call('edit', {path:'container-only.txt', edits:[{oldText:'marker',newText:'edited'}]});
assert.match(JSON.stringify(await call('bash', {command:'cat /tmp/container-only.txt; uname -s'})), /Linux/);
assert.equal((await call('bash', {command:'exit 7'})).structuredContent.exit_code, 7);
const large = await call('bash', {command:"python -c 'print(\" + JSON.stringify('x'.repeat(1)) + " * 60000)'"});
assert.equal(large.structuredContent.truncated, true);
assert.match(JSON.stringify(await call('read', {path:large.structuredContent.full_output_path})), /exceeds/);
await assert.rejects(call('bash', {command:'(sleep 2; touch /tmp/orphan) & wait', timeout:0.2}), /timed out/);
const controller = new AbortController();
setTimeout(() => controller.abort(), 200);
await assert.rejects(call('bash', {command:'(sleep 2; touch /tmp/orphan-abort) & wait'}, controller.signal), /aborted/);
await new Promise(resolve => setTimeout(resolve, 2300));
assert.equal((await call('bash', {command:'test ! -e /tmp/orphan && test ! -e /tmp/orphan-abort'})).structuredContent.exit_code, 0);
console.log('PASS: four replacements loaded; container read/write/edit/bash, exit code, timeout and abort tree cleanup');
'''.replace("ROOT", str(root)))
        subprocess.run(["node", str(script)], check=True, timeout=60,
                       env=dict(os.environ, SWE_INFER_CONTAINER=cid, SWE_INFER_WORKDIR="/tmp", EXTENSION=str(infer.EXTENSION)))
        assert not (tmp_path / "container-only.txt").exists()
    finally:
        infer.run(["docker", "rm", "-f", cid])
    assert subprocess.run(["docker", "inspect", cid], capture_output=True).returncode != 0


@pytest.mark.parametrize("routing,fallback,split,keep", [
    (True, True, False, False), (True, True, True, False),
    (True, True, False, True), (True, False, False, False),
    (False, False, False, False),
])
def test_live_fallback_fake_pi(tmp_path, capsys, routing, fallback, split, keep):
    import sys
    import time
    receipt = '{"fallback_to_main_loop": true, "reason": "pool unavailable é"}'
    events = [
        {"message": {"role": "assistant", "content": [{"type": "toolCall",
         "id": "spawn-1", "name": "darknavstar-spawn-model-pool",
         "arguments": {"category": "coding", "role": "read-write", "task": "fix"}}]}},
        {"message": {"role": "toolResult", "toolCallId": "spawn-1",
         "toolName": "darknavstar-spawn-model-pool", "content": [
             {"type": "text", "text": receipt if fallback else '{"ok": true}'}]}}]
    data = ("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n").encode()
    script = tmp_path / "fake_pi.py"
    script.write_text("import sys,time\nfrom pathlib import Path\n"
        "time.sleep(0.35)\n"
        "p=Path(sys.argv[sys.argv.index('--session')+1])\n"
        + f"data={data!r}\n"
        + ("cut=data.index('é'.encode())+1\n"
           "with p.open('wb') as f:\n"
           " f.write(data[:cut]); f.flush(); time.sleep(0.35); f.write(data[cut:]); f.flush()\n"
           if split else "p.write_bytes(data)\n")
        + ("time.sleep(30)\n" if fallback else ""))
    real_popen = subprocess.Popen
    processes, calls = [], []

    def fake_pi(command, **kwargs):
        kwargs["stdout"] = subprocess.DEVNULL  # capsys' stderr has no OS descriptor.
        process = real_popen([sys.executable, str(script), *command[1:]], **kwargs)
        processes.append(process)
        return process

    def docker(command):
        calls.append(command)
        if command[1] == "run":
            return "cid"
        if command == infer.capture_command("cid", "/testbed"):
            return "diff\n+patch\n"
        return ""

    directory = bundle(tmp_path)
    (directory / "instance.json").write_text(json.dumps([
        {"instance_id": "short", "problem_statement": "Fix"}]))
    predictions = tmp_path / "predictions.jsonl"
    started = time.monotonic()
    try:
        with patch("infer.run", side_effect=docker), patch("infer.routing_settings"), \
                patch("infer.subprocess.Popen", side_effect=fake_pi):
            status = infer.main([str(directory), "--model", "test",
                "--predictions-out", str(predictions)] +
                (["--routing"] if routing else []) + (["--keep-container"] if keep else []))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert time.monotonic() - started < 5
    captured = capsys.readouterr()
    record = json.loads(captured.out)
    trajectory = Path(record["trajectory_dir"])
    assert status == int(fallback)
    assert record["outcome"] == ("fail" if fallback else "ok")
    assert json.loads((trajectory / "run.json").read_text()) == record
    assert json.loads((tmp_path / "runs.jsonl").read_text()) == record
    assert (["docker", "rm", "-f", "cid"] in calls) is not keep
    with sqlite3.connect(tmp_path / "runs.db") as db:
        assert db.execute("SELECT outcome FROM runs").fetchone()[0] == record["outcome"]
        assert json.loads(db.execute("SELECT runs FROM tasks_short WHERE instance_id='short'").fetchone()[0]) == [trajectory.name.rsplit("__", 1)[-1]]
        if fallback:
            assert db.execute("SELECT router_child_count, missing_context_records, context_accounting_complete FROM runs").fetchone() == (0, 0, 1)
    if fallback:
        evidence = json.loads((trajectory / "fallback.json").read_text())
        assert evidence["receipt"] == receipt
        assert evidence["category"] == "coding"
        assert evidence["reason"] == "pool unavailable é"
        assert evidence["role"] == "read-write" and evidence["task"] == "fix"
        assert evidence["run_id"] == trajectory.name.rsplit("__", 1)[-1]
        assert "fallback detected (category=coding); terminating run early" in captured.err
        assert infer.capture_command("cid", "/testbed") not in calls
        assert not (trajectory / "patch.diff").exists()
        assert not predictions.exists()
        assert processes[0].returncode < 0
        assert record["total_context_tokens"] == record["main_context_tokens"]
    else:
        assert (trajectory / "patch.diff").read_text() == "diff\n+patch\n"
        assert json.loads(predictions.read_text())["model_patch"] == "diff\n+patch\n"
        assert not (trajectory / "fallback.json").exists()


def test_fallback_poller_rotation_and_invalid_lines(tmp_path):
    path = tmp_path / "session.jsonl"
    poller = infer.FallbackPoller(path)
    assert poller.poll() is None
    path.write_bytes(b'invalid\n[]\n' + b'x' * 100)
    assert poller.poll() is None
    path.write_bytes(b'{}\n')
    assert poller.poll() is None
    assert poller.offset == 3 and poller.buffer == b''
    path.unlink()
    event = {"message": {"role": "toolResult", "toolName": "darknavstar-spawn-model-pool",
        "content": [{"type": "text", "text": 'prefix {"fallback_to_main_loop":true}'}]}}
    path.write_text(json.dumps(event) + "\n")
    result = poller.poll()
    assert result["category"] is None and result["reason"] is None


def test_partial_utf8_session_metrics(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_bytes(b'{}\n{"partial":"\xc3')
    assert infer.session_entries(path) == [{}]
    assert infer.context_metrics(path, False)["main_context_tokens"] is None
