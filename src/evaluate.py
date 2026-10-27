#!/usr/bin/env python3
"""Evaluate one recorded inference run with the official SWE-bench CLI."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile

from run_tracking import migrate_evaluations, record_evaluation

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_PYTHON = Path('/Users/cc/workplace/SWE-bench/.venv/bin/python')


def repo_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def resolve_run(run_id, db_path, trajectories_dir):
    if not re.fullmatch(r'[0-9a-f]{8}', run_id):
        raise ValueError('run_id must be eight lowercase hexadecimal digits')
    if db_path.exists():
        with closing(sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
            row = db.execute('SELECT trajectory_dir, instance_id, model FROM runs WHERE run_id = ?',
                             (run_id,)).fetchone()
        if row and row[0] and repo_path(row[0]).is_dir():
            return repo_path(row[0]), row[1], row[2]
    else:
        row = None
    matches = [p for p in repo_path(trajectories_dir).glob(f'*__{run_id}') if p.is_dir()]
    if len(matches) != 1:
        raise ValueError(f'Expected one trajectory for run {run_id}; found {len(matches)}')
    directory = matches[0]
    metadata = {}
    for name in ('meta.json', 'run.json'):
        if (directory / name).exists():
            record = json.loads((directory / name).read_text(encoding='utf-8'))
            if not isinstance(record, dict):
                raise ValueError(f'Invalid run metadata: {directory / name}')
            metadata.update(record)
    return directory, (row[1] if row else metadata.get('instance_id')), (row[2] if row else metadata.get('model'))


def load_prediction(directory, instance_id, model, patch_file=None):
    """Never guess between multiple historical predictions for the same instance/model."""
    source = Path(patch_file).expanduser() if patch_file else directory / 'patch.diff'
    fallback = None
    if not source.is_file() or not model:
        predictions = directory.parent / 'predictions.jsonl'
        if predictions.exists():
            rows = [json.loads(line) for line in predictions.read_text(encoding='utf-8').splitlines() if line.strip()]
            if not all(isinstance(r, dict) for r in rows):
                raise ValueError(f'Invalid predictions: {predictions}')
            matches = [r for r in rows if r.get('instance_id') == instance_id
                       and (not model or r.get('model_name_or_path') == model)]
            if len(matches) != 1:
                raise ValueError('Historical predictions are missing or ambiguous; pass --patch FILE and record a model')
            fallback = matches[0]
    if patch_file or source.is_file():
        patch = source.read_bytes().decode('utf-8')
    elif fallback is not None:
        patch = fallback.get('model_patch')
    else:
        raise ValueError(f'No patch.diff or matching predictions for {directory}; pass --patch FILE (raw diff)')
    model = model or (fallback or {}).get('model_name_or_path')
    if not isinstance(patch, str) or not isinstance(model, str) or not model:
        raise ValueError('Prediction must contain a string patch and a nonempty model')
    return dict(instance_id=instance_id, model_name_or_path=model, model_patch=patch)


def parse_report(path, instance_id, run_id):
    report = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(report, dict) and instance_id in report:
        report = report[instance_id]
    if not isinstance(report, dict) or type(report.get('resolved')) is not bool:
        raise ValueError(f'Invalid resolved verdict in {path}')
    applied, tests = report.get('patch_successfully_applied'), report.get('tests_status')
    if (applied is not None and type(applied) is not bool) or (tests is not None and not isinstance(tests, dict)):
        raise ValueError(f'Invalid provenance in {path}')
    return dict(resolved=report['resolved'], patch_successfully_applied=applied, tests_status=tests,
                eval_run_id=run_id, report=str(path.relative_to(REPO_ROOT)),
                evaluated_at=datetime.now(timezone.utc).isoformat())


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    summary = dict(resolved=None, report=None, evaluation_stored=False)
    predictions_path = None
    keep = False
    try:
        parser = Parser(description=__doc__)
        parser.add_argument('run_id', nargs='?')
        parser.add_argument('--migrate', action='store_true', help='Convert legacy DB verdicts without evaluation')
        parser.add_argument('--runs-db', type=Path, default=REPO_ROOT / 'runs.db')
        parser.add_argument('--trajectories-dir', type=Path, default=Path('trajectories'))
        parser.add_argument('--bundle-dir', type=Path)
        parser.add_argument('--harness-python', type=Path, default=HARNESS_PYTHON)
        parser.add_argument('--timeout', type=int, default=1200)
        parser.add_argument('--keep-predictions', action='store_true')
        parser.add_argument('--patch', type=Path, help='Raw UTF-8 diff; overrides recorded patch')
        args = parser.parse_args(argv)
        if args.migrate:
            if args.run_id is not None:
                raise ValueError('--migrate does not accept a run_id')
            summary = dict(migrated=migrate_evaluations(args.runs_db))
            return 0
        if args.run_id is None:
            raise ValueError('run_id is required unless --migrate is used')
        keep = args.keep_predictions
        if args.timeout <= 0:
            raise ValueError('--timeout must be positive')
        db_path = args.runs_db.expanduser()
        directory, instance_id, model = resolve_run(args.run_id, db_path, args.trajectories_dir)
        if not isinstance(instance_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*__[A-Za-z0-9_.-]+', instance_id):
            raise ValueError('Run has no valid instance_id')
        prediction = load_prediction(directory, instance_id, model, args.patch)
        bundle = args.bundle_dir.expanduser() if args.bundle_dir else Path.home() / '.cache/swebench' / instance_id
        bundle = bundle.resolve()
        if not (bundle / 'instance.json').is_file() and not args.bundle_dir:
            # Fetch must only read an existing instance cache, never build/refresh it.
            if not (Path.home() / '.cache/swebench/instances.db').is_file():
                raise ValueError('Instance cache missing; supply --bundle-dir or fetch the bundle first')
            subprocess.run([sys.executable, str(REPO_ROOT / 'src' / 'fetch_instance.py'), instance_id,
                            str(bundle), *(['--force'] if bundle.exists() else [])], check=True, cwd=REPO_ROOT)
        rows = json.loads((bundle / 'instance.json').read_text(encoding='utf-8'))
        if (not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict)
                or rows[0].get('instance_id') != instance_id):
            raise ValueError('Bundle must contain exactly the requested instance')
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.jsonl',
                                         prefix=f'swe-eval-{args.run_id}-', delete=False) as stream:
            predictions_path = Path(stream.name)
            stream.write(json.dumps(prediction, ensure_ascii=False) + '\n')
        log_dir = REPO_ROOT / 'logs/evaluation' / args.run_id
        # A failed invocation must not persist a stale verdict from a previous patch.
        for old in log_dir.glob(f'*/{instance_id}/report.json'):
            old.unlink()
        result = subprocess.run([str(args.harness_python.expanduser()), '-m', 'swebench.harness.run_evaluation',
                                 '--dataset_name', str(bundle / 'instance.json'), '--predictions_path', str(predictions_path),
                                 '--run_id', args.run_id, '--max_workers', '1', '--timeout', str(args.timeout)], cwd=REPO_ROOT)
        reports = list(log_dir.glob(f'*/{instance_id}/report.json'))
        if len(reports) != 1:
            raise ValueError(f'Harness exit {result.returncode}: expected one report, found {len(reports)} in {log_dir}')
        evaluation = parse_report(reports[0], instance_id, args.run_id)
        verdict = 'success' if evaluation['resolved'] else 'fail'
        summary.update(resolved=evaluation['resolved'], report=evaluation['report'], verdict=verdict)
        record_evaluation(db_path, args.run_id, verdict)
        summary['evaluation_stored'] = True
        if result.returncode:
            print(f'Error: harness exited {result.returncode}', file=sys.stderr)
            return 1
        return 0
    except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    finally:
        if predictions_path:
            if keep:
                print(f'Predictions retained: {predictions_path}', file=sys.stderr)
            else:
                predictions_path.unlink(missing_ok=True)
        print(json.dumps(summary))


if __name__ == '__main__':
    sys.exit(main())
