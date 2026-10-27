#!/usr/bin/env python3
"""Run host pi against an ephemeral Docker repository (Python 3.11, stdlib only).

Pi runs in a neutral temporary directory, never the bundle or this project.
Predictions append to <bundle_dir>/../predictions.jsonl by default.
Default prompts expose only the bundle's problem_statement, never reference patches.
"""
import argparse
from datetime import datetime, timezone
import shutil
import json
import os
from pathlib import Path
import shlex
import signal
import re
import threading
import subprocess
import sys
import tempfile
import time
import uuid

from run_tracking import initialize_runs_db, append_run

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTENSION = REPO_ROOT / "pi-extension/docker-tools.ts"
RUNS_LOG = REPO_ROOT / "runs.jsonl"


def trajectory_directory(base, instance_id, routing, now=None, run_id=None):
    """Resolve relative bases against this script, never the caller's cwd."""
    base = Path(base).expanduser()
    if not base.is_absolute():
        base = REPO_ROOT / base
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", instance_id or "unknown")
    suffix = (run_id or uuid.uuid4()).hex[:8]
    return base.resolve() / f"{stamp}__{safe_id}__{'routing' if routing else 'plain'}__{suffix}"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def initialize_trajectory(directory, prompt, meta):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    write_json(directory / "meta.json", meta)


def persist_trajectory(directory, session_file, metrics, router_dir=None):
    """Preserve raw session bytes and session-linked router nodes, including descendants."""
    target = directory / "trajectory.jsonl"
    if session_file and session_file.exists():
        if session_file.resolve() != target.resolve():
            shutil.copyfile(session_file, target)
    else:
        target.touch(exist_ok=True)
    children = directory / "children"
    children.mkdir(exist_ok=True)
    session_id = None
    for line in target.read_bytes().splitlines():
        try:
            entry = json.loads(line)
            if entry.get("type") == "session":
                session_id = entry.get("id")
                break
        except (ValueError, UnicodeDecodeError):
            continue
    nodes = {}
    if metrics.get("routing") and session_id:
        for path in (router_dir or Path.home() / ".pi/model-router-trajectory").glob("*.json"):
            try:
                node = json.loads(path.read_text(encoding="utf-8"))
                nodes[node["agent_run_id"]] = (path, node)
            except (OSError, ValueError, KeyError):
                continue
    pending = [key for key, (_, node) in nodes.items() if node.get("session_id") == session_id]
    seen, child_ids = set(), set()
    while pending:
        key = pending.pop()
        if key in seen:
            continue
        seen.add(key)
        if key not in nodes:
            continue
        path, node = nodes[key]
        shutil.copyfile(path, children / path.name)
        child_ids.update(node.get("entries", {}))
        pending.extend(node.get("entries", {}))
    if not child_ids:
        (children / "README.md").write_text("no router children spawned for this run\n", encoding="utf-8")
    elif child_ids - nodes.keys():
        (children / "README.md").write_text(
            "Missing router child records: " + ", ".join(sorted(child_ids - nodes.keys())) + "\n", encoding="utf-8")
    record = dict(metrics, trajectory_dir=str(directory))
    write_json(directory / "run.json", record)
    return record


def memory_mb(value):
    """Docker's used memory, normalized to MiB (historical field suffix: _mb)."""
    match = re.fullmatch(r"\s*([\d.]+)\s*(B|KiB|MiB|GiB|TiB|kB|KB|MB|GB|TB)\s*", value.split("/")[0])
    if not match:
        raise ValueError(f"Invalid Docker memory: {value!r}")
    units = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3,
             "TiB": 1024**4, "kB": 1000, "KB": 1000, "MB": 1000**2,
             "GB": 1000**3, "TB": 1000**4}
    return float(match[1]) * units[match[2]] / 1024**2


def tree_rss_mb(output, pid):
    """Sum RSS for descendants and members of the isolated pi process group."""
    rows = []
    for line in output.splitlines():
        try:
            rows.append(tuple(map(int, line.split())))
        except ValueError:
            continue
    selected = {pid}
    while True:
        more = {p for p, parent, group, rss in rows if parent in selected or group == pid}
        if more <= selected:
            break
        selected |= more
    return sum(rss for p, parent, group, rss in rows if p in selected) / 1024


def usage_metrics(entries):
    """Last completed request input+output, separately from cumulative consumption."""
    totals = dict(input=0, output=0, cacheRead=0, cacheWrite=0, totalTokens=0)
    context = None
    for entry in entries:
        message = entry.get("message", {})
        assistant = entry.get("type") in ("message", "message_end") and message.get("role") == "assistant"
        usage = message.get("usage") if assistant else entry.get("usage")
        if not isinstance(usage, dict):
            continue
        values = {key: usage.get(key, 0) for key in totals}
        if not all(isinstance(v, (int, float)) and v >= 0 for v in values.values()):
            continue
        values["totalTokens"] = sum(values[k] for k in ("input", "output", "cacheRead", "cacheWrite"))
        for key, value in values.items():
            totals[key] += value
        if assistant and values["totalTokens"] > 0:
            context = values["totalTokens"]
    return context, totals


def session_entries(session_file):
    entries = []
    if session_file and session_file.exists():
        for line in session_file.read_bytes().splitlines():
            try:
                entry = json.loads(line)
                if isinstance(entry, dict):
                    entries.append(entry)
            except (ValueError, UnicodeDecodeError):
                pass  # A cancelled process can leave an incomplete trailing line.
    return entries


def router_spawn_fallback_receipt(entries):
    """Inspect only main-session spawn results, not calls or child records."""
    def contains_fallback(value):
        if isinstance(value, dict):
            return value.get("fallback_to_main_loop") is True or any(
                contains_fallback(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_fallback(item) for item in value)
        return False

    for entry in entries:
        message = entry.get("message", {})
        if (message.get("role") != "toolResult" or
                message.get("toolName") != "darknavstar-spawn-model-pool"):
            continue
        for block in message.get("content", []):
            if block.get("type") != "text":
                continue
            text = block.get("text", "")
            try:
                if contains_fallback(json.loads(text)):
                    return text
            except ValueError:
                if re.search(r'"fallback_to_main_loop"\s*:\s*true\b', text):
                    return text
    return None


def router_spawn_fallback(entries):
    return router_spawn_fallback_receipt(entries) is not None


class FallbackPoller:
    """Tail complete binary JSONL lines; retain split UTF-8 until the LF arrives."""
    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.buffer = b""
        self.identity = None
        self.calls = {}

    def poll(self):
        try:
            with self.path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if identity != self.identity or stat.st_size < self.offset:
                    self.offset, self.buffer = 0, b""
                    self.calls.clear()
                self.identity = identity
                stream.seek(self.offset)
                data = stream.read()
                self.offset += len(data)
        except FileNotFoundError:
            return None
        lines = (self.buffer + data).split(b"\n")
        self.buffer = lines.pop()
        for line in lines:
            try:
                entry = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(entry, dict):
                continue
            message = entry.get("message", {})
            if message.get("role") == "assistant":
                for block in message.get("content", []):
                    if (block.get("type") == "toolCall" and
                            block.get("name") == "darknavstar-spawn-model-pool"):
                        self.calls[block.get("id")] = block.get("arguments", {})
            receipt = router_spawn_fallback_receipt([entry])
            if receipt is not None:
                args = self.calls.get(message.get("toolCallId"), {})
                if not isinstance(args, dict):
                    args = {}

                def reason(value):
                    if isinstance(value, dict):
                        if "reason" in value:
                            return value["reason"]
                        values = value.values()
                    elif isinstance(value, list):
                        values = value
                    else:
                        return None
                    return next((r for v in values if (r := reason(v)) is not None), None)

                try:
                    cause = reason(json.loads(receipt))
                except ValueError:
                    cause = None
                return dict(receipt=receipt, reason=cause,
                            **{key: args.get(key) for key in ("category", "role", "task")})
        return None


def context_metrics(session_file, routing, trajectory=None):
    entries = session_entries(session_file)
    main, usage = usage_metrics(entries)
    total, children, missing = main, 0, 0
    session_id = next((e["id"] for e in entries if e.get("type") == "session"), None)
    nodes = {}
    if routing and session_id:
        for path in (trajectory or Path.home() / ".pi/model-router-trajectory").glob("*.json"):
            try:
                node = json.loads(path.read_text())
                nodes[node["agent_run_id"]] = node
            except (OSError, ValueError, KeyError):
                continue
        root = next((n for n in nodes.values() if n.get("session_id") == session_id), None)
        pending = list(root.get("entries", {})) if root else []
        missing += int(root is None)
        seen = set()
        while pending:
            child_id = pending.pop()
            if child_id in seen:
                continue
            seen.add(child_id)
            children += 1
            child = nodes.get(child_id, {})
            ctx, tokens = usage_metrics((child.get("detail") or {}).get("events", []))
            missing += int(ctx is None)
            if ctx is not None:
                total = (total or 0) + ctx
            for key in usage:
                usage[key] += tokens[key]
            pending.extend(child.get("entries", {}))
    elif routing:
        missing += 1
    return dict(main_context_tokens=main, total_context_tokens=total,
                token_usage=usage, router_child_count=children,
                context_accounting_complete=main is not None and missing == 0,
                missing_context_records=missing)


class ResourceSampler:
    """Best-effort sampling off the agent's critical path; CLI calls are bounded."""
    def __init__(self, container, pid):
        self.container, self.pid = container, pid
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.metrics = dict(container_mem_peak_mb=None, container_mem_final_mb=None,
                            host_pi_mem_peak_mb=None, resource_sample_errors=0)

    def sample(self):
        for command, key, parse in [
            (["ps", "-axo", "pid=,ppid=,pgid=,rss="], "host_pi_mem_peak_mb",
             lambda s: tree_rss_mb(s, self.pid)),
            (["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", self.container],
             "container_mem_peak_mb", memory_mb),
        ]:
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=4)
                value = round(parse(result.stdout), 3)
                self.metrics[key] = max(self.metrics[key] or 0, value)
                if key == "container_mem_peak_mb":
                    self.metrics["container_mem_final_mb"] = value
            except (OSError, ValueError, subprocess.SubprocessError):
                self.metrics["resource_sample_errors"] += 1

    def loop(self):
        while not self.stop.is_set():
            started = time.monotonic()
            self.sample()
            self.stop.wait(max(0, 2 - (time.monotonic() - started)))

    def finish(self):
        self.stop.set()
        self.thread.join()
        self.sample()


def read_bundle(bundle):
    image = (bundle / "image.txt").read_text().rstrip("\r\n")
    if not image or image.startswith("-") or any(c.isspace() for c in image):
        raise ValueError("Invalid image.txt reference")
    rows = json.loads((bundle / "instance.json").read_text())
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0].get("instance_id"), str):
        raise ValueError("instance.json must contain one row with instance_id")
    return image, rows[0]["instance_id"]


def docker_run(image, name, workdir):
    return ["docker", "run", "-d", "--init", "-it", "--name", name, "-w", workdir, image, "sleep", "infinity"]


def docker_exec(container, *command):
    return ["docker", "exec", container, *command]


def capture_command(container, workdir):
    return docker_exec(container, "bash", "-lc", f"cd {shlex.quote(workdir)} && git add -N . && git diff --no-ext-diff")


def routing_settings(directory):
    """Overlay user configuration without modifying it; children inherit the directory.

    Pi has no extra-extension environment variable. PI_CODING_AGENT_DIR plus
    settings.extensions is the supported cross-process resource-loading seam.
    """
    source = Path(os.environ.get("PI_CODING_AGENT_DIR", "~/.pi/agent")).expanduser().resolve()
    directory = Path(directory)
    directory.mkdir()
    if source.exists():
        for entry in source.iterdir():
            if entry.name != "settings.json":
                (directory / entry.name).symlink_to(entry)
    settings_file = source / "settings.json"
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}

    def absolute(value):
        prefix = value[:1] if value.startswith(("!", "+", "-")) else ""
        path = value[len(prefix):]
        if path.startswith(("builtin:", "npm:", "git:", "https://", "http://")):
            return value
        return prefix + str((source / Path(path).expanduser()).absolute())

    for key in ("extensions", "skills", "prompts", "themes"):
        settings[key] = [absolute(value) for value in settings.get(key, [])]
    settings["packages"] = [
        dict(package, source=absolute(package["source"])) if isinstance(package, dict)
        else absolute(package) for package in settings.get("packages", [])]
    # A plain path discovers the file; +path only enables an already discovered
    # resource and cannot load an otherwise unknown extension by itself.
    settings["extensions"].extend([str(EXTENSION), "+" + str(EXTENSION)])
    settings["defaultTools"] = ["bash", "read", "edit", "write"]
    # Children do not inherit --no-mcp; disable the built-in through settings too.
    settings["extensions"].append("-builtin:mcp")
    (directory / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")


def pi_command(model, prompt, routing=False):
    # --tools is an allowlist for extension tools too, not just built-ins.
    tools = "bash,read,edit,write"
    if routing:
        tools += (",darknavstar-get-model-pool,darknavstar-spawn-model-pool"
                  ",darknavstar-get-agent-run-records")
    # Routing keeps user packages enabled; the inherited overlay wraps children.
    return ["pi", "--print", "--no-session", *([] if routing else ["--no-extensions"]), "--no-mcp",
            "--no-skills", "--no-prompt-templates", "--no-context-files", "--no-approve",
            "--tools", tools, "--extension", str(EXTENSION),
            "--model", model, "--", prompt]


def prediction_line(instance_id, model, patch):
    return json.dumps(dict(instance_id=instance_id, model_name_or_path=model,
                           model_patch=patch), ensure_ascii=False) + "\n"


def run(command):
    return subprocess.run(command, check=True, stdout=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace").stdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--thinking-level", help="Forward an explicit thinking level to pi")
    parser.add_argument("--workdir", default="/testbed")
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--routing", action="store_true",
                        help="Enable user model-router package; wrap child tools via a temporary Pi config")
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--runs-db", type=Path, default=REPO_ROOT / "runs.db",
                        help="SQLite run tracking database (default: repo-root runs.db)")
    parser.add_argument("--trajectories-dir", type=Path, default=Path("trajectories"),
                        help="Persistent run records (relative paths resolve against the repo root)")
    args = parser.parse_args(argv)
    if not args.dry_run:
        try:
            initialize_runs_db(args.runs_db.expanduser(), RUNS_LOG)
        except Exception as exc:
            print(f"Error initializing runs database: {exc}", file=sys.stderr)
            return 1
    started = time.monotonic()
    instance_id, patch, outcome = None, "", "error"
    name = "swebench-infer-" + uuid.uuid4().hex
    container = None
    attempted = False
    db_error = False
    old_handlers = {}
    sampler, session_file = None, None
    trajectory_dir = None
    early_fallback = None
    prompt = args.prompt or ""
    meta = dict(instance_id=None, model=args.model, container_id=None, image=None,
                head_commit=None, git_status_at_start=None)
    try:
        image, instance_id = read_bundle(args.bundle_dir)
        if not args.workdir.startswith("/"):
            raise ValueError("--workdir must be absolute")
        if args.prompt:
            prompt = args.prompt
        else:
            row = json.loads((args.bundle_dir / "instance.json").read_text(encoding="utf-8"))[0]
            issue = row.get("problem_statement")
            if not isinstance(issue, str) or not issue.strip():
                raise ValueError("Bundle has no problem_statement; supply --prompt")
            prompt = (f"You are in a repository at {args.workdir}. Solve the task described in the issue. "
                      f"Make your changes in {args.workdir}. Do not commit.\n\n" + issue)
        # Avoid pi's @file expansion; prompts are always literal task text.
        if prompt.startswith("@"):
            raise ValueError("Prompt must not start with @ (pi file expansion)")
        output = args.predictions_out or args.bundle_dir.parent / "predictions.jsonl"
        launch = docker_run(image, name, args.workdir)
        status = docker_exec(name, "git", "-C", args.workdir, "status", "--porcelain")
        head = docker_exec(name, "git", "-C", args.workdir, "rev-parse", "HEAD")
        agent = pi_command(args.model, prompt, args.routing)
        if args.thinking_level:
            agent[agent.index("--"):agent.index("--")] = ["--thinking", args.thinking_level]
        trajectory_dir = trajectory_directory(args.trajectories_dir, instance_id, args.routing)
        session_file = trajectory_dir / "trajectory.jsonl"
        agent[agent.index("--no-session"):agent.index("--no-session") + 1] = ["--session", str(session_file)]
        meta.update(instance_id=instance_id, image=image)
        if args.dry_run:
            print(f"# trajectory dir: {trajectory_dir}")
            print("# pi cwd: fresh neutral temporary directory outside bundle; no mounts")
            for command in [launch, status, head]:
                print(shlex.join(command))
            if args.routing:
                print("# create temporary agent-dir overlay: user settings/packages + docker-tools.ts; inherited by children")
            routing_env = 'PI_CODING_AGENT_DIR="$PI_RUN_AGENT_DIR" ' if args.routing else ""
            print(routing_env + f"SWE_INFER_CONTAINER={name} SWE_INFER_WORKDIR={shlex.quote(args.workdir)} " + shlex.join(agent))
            print(shlex.join(capture_command(name, args.workdir)))
            print(f"# append prediction to {output}")
            if not args.keep_container:
                print(shlex.join(["docker", "rm", "-f", name]))
            return 0

        def cancel(signum, frame):
            raise KeyboardInterrupt
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, cancel)
        initialize_trajectory(trajectory_dir, prompt, meta)
        try:
            run(["docker", "image", "inspect", image])
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Docker image {image!r} is not available locally. Pull it first with "
                f"fetch_instance.py {instance_id} or "
                f"docker pull --platform linux/amd64 {image}"
            ) from exc
        attempted = True
        container = run(launch).strip()
        meta["container_id"] = container
        meta["git_status_at_start"] = run(status)
        meta["head_commit"] = run(head).strip()
        write_json(trajectory_dir / "meta.json", meta)
        if meta["git_status_at_start"].strip():
            raise ValueError("Container repository is not clean")
        print("HEAD: " + meta["head_commit"], file=sys.stderr)
        env = dict(os.environ, SWE_INFER_CONTAINER=container, SWE_INFER_WORKDIR=args.workdir)
        # Do not inherit the calling harness's session/router identity.
        for key in ("PI_SESSION_FILE", "PI_MODEL_ROUTER_CHILD_CONFIG", "PI_MODEL_ROUTER_RUN_ID",
                    "PI_MODEL_ROUTER_PARENT_RUN_ID"):
            env.pop(key, None)
        with tempfile.TemporaryDirectory(prefix="swebench-pi-") as cwd:
            if args.routing:
                agent_dir = Path(cwd) / "agent"
                routing_settings(agent_dir)
                env["PI_CODING_AGENT_DIR"] = str(agent_dir)
            # Separate process group: run cancellation snapshots diff before container teardown.
            process = subprocess.Popen(agent, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                       stdout=sys.stderr, start_new_session=True)
            sampler = ResourceSampler(container, process.pid)
            sampler.thread.start()
            try:
                poller = FallbackPoller(session_file) if args.routing else None
                deadline = time.monotonic() + 1500
                while True:
                    if poller:
                        early_fallback = poller.poll()
                        if early_fallback is not None:
                            write_json(trajectory_dir / "fallback.json", dict(
                                run_id=trajectory_dir.name.rsplit("__", 1)[-1],
                                detected_at=datetime.now(timezone.utc).isoformat(),
                                **early_fallback))
                            print(f"fallback detected (category={early_fallback['category']}); "
                                  "terminating run early", file=sys.stderr, flush=True)
                            break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(agent, 1500)
                    try:
                        code = process.wait(timeout=min(0.2, remaining) if poller else remaining)
                    except subprocess.TimeoutExpired:
                        if poller:
                            continue
                        raise
                    if code != 0:
                        raise RuntimeError("pi exited unsuccessfully")
                    break
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        outcome = "ok"
    except KeyboardInterrupt:
        outcome = "cancelled"
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
    finally:
        if not args.dry_run:
            # Ignore repeated cancellation while preserving the patch and releasing resources.
            for sig in old_handlers:
                signal.signal(sig, signal.SIG_IGN)
            if sampler:
                sampler.finish()
            if attempted:
                try:
                    if early_fallback is None:
                        patch = run(capture_command(container or name, args.workdir))
                        (trajectory_dir / "patch.diff").write_bytes(patch.encode("utf-8"))
                        output.parent.mkdir(parents=True, exist_ok=True)
                        with output.open("a", encoding="utf-8") as stream:
                            stream.write(prediction_line(instance_id, args.model, patch))
                except Exception as exc:
                    print(f"Error capturing prediction: {exc}", file=sys.stderr)
                    if outcome != "cancelled":
                        outcome = "error"
                finally:
                    if not args.keep_container:
                        try:
                            run(["docker", "rm", "-f", container or name])
                        except Exception as exc:
                            print(f"Error removing container: {exc}", file=sys.stderr)
                            if outcome != "cancelled":
                                outcome = "error"
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            metrics = dict(instance_id=instance_id, outcome=outcome, routing=args.routing,
                           duration_seconds=round(time.monotonic() - started, 3),
                           model_patch_bytes=len(patch.encode("utf-8")),
                           model=args.model, session_file=str(session_file) if session_file else None)
            metrics.update(sampler.metrics if sampler else dict(container_mem_peak_mb=None,
                           container_mem_final_mb=None, host_pi_mem_peak_mb=None, resource_sample_errors=0))
            metrics.update(context_metrics(session_file, args.routing and early_fallback is None))
            if early_fallback is not None or router_spawn_fallback(session_entries(session_file)):
                outcome = metrics["outcome"] = "fail"
                # User rule: ANY fallback zeros counted children, even when other
                # spawns succeeded. Preserve token usage and raw child evidence.
                metrics.update(router_child_count=0, missing_context_records=0,
                               context_accounting_complete=True,
                               total_context_tokens=metrics["main_context_tokens"])
            if trajectory_dir is None:
                trajectory_dir = trajectory_directory(args.trajectories_dir, instance_id, args.routing)
            if not trajectory_dir.exists():
                initialize_trajectory(trajectory_dir, prompt, meta)
            write_json(trajectory_dir / "meta.json", meta)
            metrics = persist_trajectory(trajectory_dir, session_file, metrics)
            record = json.dumps(metrics)
            with RUNS_LOG.open("a", encoding="utf-8") as stream:
                stream.write(record + "\n")
            print(record)
            try:
                append_run(args.runs_db.expanduser(), metrics)
            except Exception as exc:
                print(f"Error recording run in SQLite: {exc}", file=sys.stderr)
                db_error = True
    if db_error:
        return 1
    return 0 if outcome == "ok" else (130 if outcome == "cancelled" else 1)


if __name__ == "__main__":
    sys.exit(main())
