"""Fetch once, infer in isolation in parallel, evaluate after the barrier, then remove the image."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_FILES = ('instance.json', 'eval.sh', 'gold.patch', 'test_patch.diff', 'image.txt')


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(config, dict) or set(config) != {'inferences'}:
        raise ValueError('Config must contain only inferences')
    entries = config['inferences']
    if not isinstance(entries, list) or not entries:
        raise ValueError('inferences must be a nonempty list')
    labels = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {'label', 'model', 'routing', 'thinking_level'}:
            raise ValueError('Unknown inference keys or invalid entry')
        label, model = entry.get('label'), entry.get('model')
        if not isinstance(label, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', label):
            raise ValueError('label must be filesystem-safe (1-80 letters, digits, underscores or hyphens)')
        if label.casefold() in labels:
            raise ValueError('Duplicate label')
        labels.add(label.casefold())
        if not isinstance(model, str) or not model.strip():
            raise ValueError('model is required')
        if type(entry.get('routing', False)) is not bool:
            raise ValueError('routing must be boolean')
        thinking = entry.get('thinking_level')
        if thinking is not None and (not isinstance(thinking, str) or not thinking.strip()):
            raise ValueError('thinking_level must be a nonempty string or null')
    return entries


def fan_out(staging, work, entries):
    for entry in entries:
        bundle = work / ('run-' + entry['label']) / 'bundle'
        bundle.mkdir(parents=True)
        for name in BUNDLE_FILES:
            shutil.copy2(staging / name, bundle / name)


def inference_command(entry, directory, db, trajectories):
    command = [sys.executable, str(REPO_ROOT / 'src' / 'infer.py'), str(directory / 'bundle'),
               '--model', entry['model'], '--predictions-out', str(directory / 'predictions.jsonl'),
               '--runs-db', str(db), '--trajectories-dir', str(trajectories)]
    if entry.get('routing'):
        command.append('--routing')
    if entry.get('thinking_level'):
        command.extend(['--thinking-level', entry['thinking_level']])
    return command


def run_command(cmd, log_path, timeout=None):
    """Keep complete output on disk, replay to stderr at exit; stdout remains JSON summaries.

    On timeout send SIGTERM to the runner, then wait for its patch/container cleanup.
    Inference's own agent wait is bounded; do not SIGKILL it during teardown.
    """
    started = time.monotonic()
    timed_out = False
    with Path(log_path).open('w', encoding='utf-8') as log:
        log.write('$ ' + shlex.join(cmd) + '\n')
        log.flush()
        with subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT) as process:
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.terminate()
                process.wait()
                code = 124
    output = Path(log_path).read_text(encoding='utf-8', errors='replace')
    print(output, file=sys.stderr, flush=True)
    print(json.dumps(dict(command=cmd, seconds=round(time.monotonic() - started, 3),
                          exit=code, timed_out=timed_out)), file=sys.stderr, flush=True)
    return subprocess.CompletedProcess(cmd, code, output)


def execute(command, log, timeout):
    try:
        return run_command(command, log, timeout=timeout)
    except Exception as exc:
        message = f'{type(exc).__name__}: {exc}'
        print(message, file=sys.stderr)
        try:
            with Path(log).open('a', encoding='utf-8') as stream:
                stream.write(message + '\n')
        except OSError:
            pass
        return subprocess.CompletedProcess(command, 1, message)


def identify_record(output, instance, entry):
    # infer prints the same final record it appends to runs.jsonl. A private log
    # avoids races with unrelated invocations or same-model/same-routing entries.
    for line in reversed(output.splitlines()):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if (isinstance(record, dict) and record.get('instance_id') == instance
                and record.get('model') == entry['model'] and record.get('trajectory_dir')
                and 'duration_seconds' in record and 'outcome' in record):
            return record
    return None


def read_run(db, run_id):
    if not db.exists() or run_id is None:
        return None
    with closing(sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
    return dict(row) if row else None


def evaluation_command(run, db, timeout=None):
    command = [sys.executable, str(REPO_ROOT / 'src' / 'evaluate.py'), run['run_id'],
               '--bundle-dir', str(run['directory'] / 'bundle'), '--runs-db', str(db)]
    if timeout is not None:
        command.extend(['--timeout', str(math.ceil(timeout))])
    return command


def summarize(run, db):
    row = read_run(db, run['run_id'])
    verdict = row['evaluation'] if row else None
    reports = list((REPO_ROOT / 'logs/evaluation' / run['run_id']).glob('*/*/report.json')) if run['run_id'] else []
    return dict(label=run['entry']['label'], run_id=run['run_id'],
                trajectory_dir=run['trajectory_dir'], inference_exit=run['result'].returncode,
                outcome=run['outcome'], resolved=(verdict == 'success') if verdict in ('success', 'fail') else None,
                report=str(reports[0].relative_to(REPO_ROOT)) if len(reports) == 1 else None,
                evaluated=(run['evaluation_exit'] == 0) if run['evaluation_exit'] is not None else None,
                evaluation_exit=run['evaluation_exit'], error=run.get('error'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Retains private bundles and logs under /tmp/swe-orchestrate/<UTC>__<instance>__<unique>/. '
        'Timeout bounds inference waiting and harness tests; cleanup is awaited before image removal.'))
    parser.add_argument('instance_id')
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'config.json')
    parser.add_argument('--keep-image', action='store_true')
    parser.add_argument('--runs-db', type=Path, default=REPO_ROOT / 'runs.db')
    parser.add_argument('--trajectories-dir', type=Path, default=Path('trajectories'))
    parser.add_argument('--timeout', type=float,
                        help='Inference wait / evaluation test seconds (default: child limits; fetch and cleanup unbounded)')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing or creating files')
    args = parser.parse_args(argv)
    image, removed, work, runs = None, None, None, []
    removal = 'not attempted'
    status = 0
    try:
        entries = load_config(args.config.expanduser())
        if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
            raise ValueError('--timeout must be positive')
        db = args.runs_db.expanduser().resolve()
        trajectories = args.trajectories_dir.expanduser()
        trajectories = (REPO_ROOT / trajectories).resolve()
        safe_id = re.sub(r'[^A-Za-z0-9_-]', '_', args.instance_id)
        prefix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '__' + safe_id[:100] + '__'
        base = Path('/tmp/swe-orchestrate')
        if args.dry_run:
            work = base / (prefix + 'DRY-RUN')
        else:
            base.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
        staging = work / 'staging'
        fetch = [sys.executable, str(REPO_ROOT / 'src' / 'fetch_instance.py'), args.instance_id, str(staging)]
        if args.dry_run:
            print(shlex.join(fetch))
            for entry in entries:
                directory = work / ('run-' + entry['label'])
                print(f'# copy five staging bundle files to {directory / "bundle"}')
                print(shlex.join(inference_command(entry, directory, db, trajectories)))
            print('# wait for ALL inferences, then evaluate patch-present runs in parallel')
            for entry in entries:
                print(shlex.join(evaluation_command(dict(run_id='<run_id:' + entry['label'] + '>',
                      directory=work / ('run-' + entry['label'])), db, args.timeout)))
            if not args.keep_image:
                print('docker rmi <image-from-staging/image.txt>')
            return 0
        result = execute(fetch, work / 'fetch.log', None)
        if (staging / 'image.txt').exists():
            image = (staging / 'image.txt').read_text().strip()
        if result.returncode:
            raise RuntimeError(f'Fetch failed (exit {result.returncode}); see {work / "fetch.log"}')
        if not image:
            raise ValueError('Fetch produced no image reference')
        fan_out(staging, work, entries)
        with ThreadPoolExecutor(max_workers=len(entries)) as pool:
            jobs = []
            for entry in entries:
                directory = work / ('run-' + entry['label'])
                jobs.append((entry, directory, pool.submit(execute,
                    inference_command(entry, directory, db, trajectories), directory / 'infer.log', args.timeout)))
            for entry, directory, job in jobs:
                result = job.result()
                record = identify_record(result.stdout, args.instance_id, entry)
                trajectory = record['trajectory_dir'] if record else None
                run_id = Path(trajectory).name.rsplit('__', 1)[-1] if trajectory else None
                row = read_run(db, run_id)
                if row:
                    trajectory = row['trajectory_dir']
                runs.append(dict(entry=entry, directory=directory, result=result, run_id=run_id,
                    trajectory_dir=trajectory, outcome=record['outcome'] if record else 'error',
                    evaluation_exit=None, error=(f'Inference exit {result.returncode}; see {directory / "infer.log"}'
                                                if result.returncode else None)))
        # The inference executor is fully drained before any evaluator is submitted.
        with ThreadPoolExecutor(max_workers=len(entries)) as pool:
            jobs = []
            for run in runs:
                if run['outcome'] in ('fail', 'error'):
                    continue  # Failed/error runs retain patches but are not evaluated.
                if run['trajectory_dir'] and (Path(run['trajectory_dir']) / 'patch.diff').is_file():
                    # Let the official harness enforce its test timeout and clean up;
                    # killing evaluate.py would orphan its harness/container.
                    jobs.append((run, pool.submit(execute, evaluation_command(run, db, args.timeout),
                                                 run['directory'] / 'evaluate.log', None)))
                elif not run['error']:
                    run['error'] = 'No captured patch or final inference record'
            for run, job in jobs:
                run['evaluation_exit'] = job.result().returncode
                if run['evaluation_exit']:
                    run['error'] = (run['error'] or '') + f' Evaluation failed; see {run["directory"] / "evaluate.log"}'
    except Exception as exc:
        print(f'Orchestration error: {exc}', file=sys.stderr)
        status = 1
    finally:
        if not args.dry_run and image:
            if args.keep_image:
                removal = 'kept by request'
            else:
                result = execute(['docker', 'rmi', image], work / 'image-removal.log', None)
                absent = 'no such image' in result.stdout.lower()
                removed = result.returncode == 0 or absent
                removal = 'already absent' if absent else ('removed' if removed else result.stdout.strip())
                if not removed:
                    status = 1
    if args.dry_run:
        return status
    for run in runs:
        try:
            print(json.dumps(summarize(run, db)), flush=True)
        except Exception as exc:
            print(f'Summary error: {exc}', file=sys.stderr)
            status = 1
    print(json.dumps(dict(instance_id=args.instance_id, image=image, image_removed=removed,
                          image_removal=removal, work_dir=str(work) if work else None)), flush=True)
    return status


if __name__ == '__main__':
    sys.exit(main())
