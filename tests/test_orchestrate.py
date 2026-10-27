"""Offline orchestration tests; subprocesses are mocked, no Docker or network."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest
import orchestrate as o


@pytest.fixture
def entries():
    return [dict(label='plain', model='provider/model', routing=False),
            dict(label='routing', model='provider/model', routing=True, thinking_level='medium')]


@pytest.mark.parametrize('verdict,resolved', [('success', True), ('fail', False), (None, None)])
def test_summary_verdict(tmp_path, monkeypatch, verdict, resolved):
    monkeypatch.setattr(o, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(o, 'read_run', lambda db, run_id: {'evaluation': verdict})
    report = tmp_path / 'logs/evaluation/1234abcd/model/instance/report.json'
    report.parent.mkdir(parents=True)
    report.write_text('{}')
    run = dict(entry={'label': 'plain'}, run_id='1234abcd', trajectory_dir='trajectory',
               result=subprocess.CompletedProcess([], 0), outcome='ok', evaluation_exit=0)
    summary = o.summarize(run, tmp_path / 'runs.db')
    assert summary['resolved'] is resolved
    assert summary['report'] == str(report.relative_to(tmp_path))


def test_config(tmp_path, entries):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(dict(inferences=entries)))
    assert o.load_config(path) == entries


@pytest.mark.parametrize('entries', [[], [{'label': 'x'}],
    [{'label': 'x', 'model': 'm'}, {'label': 'x', 'model': 'm'}],
    [{'label': 'x', 'model': 'm', 'unknown': 1}], [{'label': '../bad', 'model': 'm'}],
    [{'label': 'x', 'model': 'm', 'routing': 'yes'}],
    [{'label': 'x', 'model': 'm', 'thinking_level': 3}]])
def test_invalid_config(tmp_path, entries):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(dict(inferences=entries)))
    with pytest.raises(ValueError):
        o.load_config(path)


def test_unknown_top_key(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text('{"inferences": [], "extra": true}')
    with pytest.raises(ValueError):
        o.load_config(path)


def test_fan_out(tmp_path, entries):
    staging = tmp_path / 'staging'
    staging.mkdir()
    for name in o.BUNDLE_FILES:
        (staging / name).write_bytes(b'content\r\n\xff')
    (staging / 'eval.sh').chmod(0o755)
    o.fan_out(staging, tmp_path, entries)
    for entry in entries:
        bundle = tmp_path / ('run-' + entry['label']) / 'bundle'
        for name in o.BUNDLE_FILES:
            assert (bundle / name).read_bytes() == (staging / name).read_bytes()
            assert (bundle / name).stat().st_ino != (staging / name).stat().st_ino
        assert (bundle / 'eval.sh').stat().st_mode & 0o111
    (tmp_path / 'run-plain/bundle/gold.patch').write_text('changed')
    assert (tmp_path / 'run-routing/bundle/gold.patch').read_bytes() == b'content\r\n\xff'


def test_commands(tmp_path, entries):
    base = [sys.executable, str(o.REPO_ROOT / 'src' / 'infer.py'), str(tmp_path / 'bundle'),
            '--model', 'provider/model', '--predictions-out', str(tmp_path / 'predictions.jsonl'),
            '--runs-db', 'db', '--trajectories-dir', 'traj']
    assert o.inference_command(entries[0], tmp_path, Path('db'), Path('traj')) == base
    assert o.inference_command(entries[1], tmp_path, Path('db'), Path('traj')) == (
        base + ['--routing', '--thinking-level', 'medium'])
    assert o.evaluation_command(dict(run_id='1234abcd', directory=tmp_path), Path('db')) == [
        sys.executable, str(o.REPO_ROOT / 'src' / 'evaluate.py'), '1234abcd',
        '--bundle-dir', str(tmp_path / 'bundle'), '--runs-db', 'db']


@pytest.mark.parametrize('fallback,partial_patch', [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize('keep,missing', [(False, False), (True, False), (False, True)])
def test_failure_isolation(tmp_path, monkeypatch, capsys, entries, keep, missing, fallback, partial_patch):
    monkeypatch.setattr(o, 'REPO_ROOT', tmp_path)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(inferences=entries)))
    db = tmp_path / 'runs.db'
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.execute('CREATE TABLE runs (run_id TEXT, trajectory_dir TEXT, evaluation TEXT)')
    barrier = threading.Barrier(2)
    completed, commands = [], []
    def run(cmd, log_path, timeout=None):
        commands.append(cmd)
        output, code = '', 0
        if cmd[0] == 'docker':
            assert len(completed) == 2
            assert any('evaluate.py' in c[1] for c in commands if len(c) > 1)
            output, code = ('Error: No such image: example', 1) if missing else ('deleted', 0)
        elif cmd[1].endswith('fetch_instance.py'):
            staging = Path(cmd[3])
            staging.mkdir()
            for name in o.BUNDLE_FILES:
                (staging / name).write_text('example')
        elif cmd[1].endswith('infer.py'):
            barrier.wait(timeout=5)  # Fails if inference is accidentally serialized.
            routing = '--routing' in cmd
            run_id = '1234abcd' if routing else '0000abcd'
            directory = tmp_path / ('trajectories/time__' + run_id)
            directory.mkdir(parents=True)
            if routing or partial_patch:
                (directory / 'patch.diff').write_bytes(b'partial patch\n')
            with closing(sqlite3.connect(db)) as connection, connection:
                connection.execute('INSERT INTO runs VALUES (?, ?, NULL)', (run_id, str(directory)))
            record = dict(instance_id='test__issue-1', model='provider/model', trajectory_dir=str(directory),
                          duration_seconds=1, outcome='ok' if routing else ('fail' if fallback else 'error'))
            output = 'noise\n' + json.dumps(record) + '\n'
            code = 0 if routing else 1
            completed.append(run_id)
        else:
            assert len(completed) == 2  # Evaluation phase barrier.
            assert cmd[2] == '1234abcd'
            with closing(sqlite3.connect(db)) as connection, connection:
                connection.execute('UPDATE runs SET evaluation=? WHERE run_id=?',
                                   ('fail', cmd[2]))
        Path(log_path).write_text(output)
        return subprocess.CompletedProcess(cmd, code, output)
    monkeypatch.setattr(o, 'run_command', run)
    assert o.main(['test__issue-1', '--config', str(config), '--runs-db', str(db)] +
                  (['--keep-image'] if keep else [])) == 0
    summaries = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    plain, routed, aggregate = summaries
    assert plain['inference_exit'] == 1 and plain['evaluated'] is None and plain['error']
    assert plain['outcome'] == ('fail' if fallback else 'error')
    evaluations = [c for c in commands if len(c) > 1 and c[1].endswith('evaluate.py')]
    assert [c[2] for c in evaluations] == ['1234abcd']
    with closing(sqlite3.connect(db)) as connection:
        assert connection.execute(
            'SELECT evaluation FROM runs WHERE run_id=?', ('0000abcd',)).fetchone() == (None,)
    assert routed['resolved'] is False and routed['report'] is None
    assert routed['evaluated'] is True
    assert aggregate['image_removed'] is (None if keep else True)
    if missing:
        assert aggregate['image_removal'] == 'already absent'
    assert sum(c[0] == 'docker' for c in commands) == (0 if keep else 1)
    assert sum(c[1].endswith('fetch_instance.py') for c in commands) == 1


def test_missing_record_and_db(tmp_path):
    assert o.identify_record('not json\n{}', 'i', dict(model='m')) is None
    assert o.read_run(tmp_path / 'missing.db', '1234abcd') is None
    assert not (tmp_path / 'missing.db').exists()


def test_fetch_failure(tmp_path, monkeypatch, capsys, entries):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(inferences=entries)))
    monkeypatch.setattr(o, 'run_command', lambda *a, **kw: subprocess.CompletedProcess(a[0], 1, 'not found'))
    assert o.main(['missing', '--config', str(config)]) == 1
    assert 'Fetch failed' in capsys.readouterr().err


def test_dry_run(tmp_path, monkeypatch, capsys, entries):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(inferences=entries)))
    def unexpected(*a, **kw):
        pytest.fail('dry-run executed a command')
    monkeypatch.setattr(o, 'run_command', unexpected)
    assert o.main(['test__issue-1', '--config', str(config), '--dry-run']) == 0
    output = capsys.readouterr().out
    assert '--thinking-level medium' in output and 'docker rmi' in output
    assert not (tmp_path / 'runs.db').exists()


def test_thinking_forwarded(tmp_path, capsys):
    import infer
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'image.txt').write_text('example')
    (bundle / 'instance.json').write_text('[{"instance_id":"test__issue-1","problem_statement":"Fix it"}]')
    assert infer.main([str(bundle), '--model', 'provider/model', '--thinking-level', 'medium', '--dry-run']) == 0
    output = capsys.readouterr().out
    assert '--thinking medium -- ' in output


def test_null_thinking_and_case_collision(tmp_path):
    path = tmp_path / 'config.json'
    entry = dict(label='Plain', model='m', thinking_level=None)
    path.write_text(json.dumps(dict(inferences=[entry])))
    assert o.load_config(path) == [entry]
    path.write_text(json.dumps(dict(inferences=[entry, dict(label='plain', model='m')])))
    with pytest.raises(ValueError, match='Duplicate'):
        o.load_config(path)


def test_command_logging(tmp_path, capsys):
    log = tmp_path / 'command.log'
    result = o.run_command([sys.executable, '-c', 'print("child output")'], log)
    assert result.returncode == 0
    assert 'child output' in log.read_text()
    captured = capsys.readouterr()
    assert captured.out == '' and 'child output' in captured.err


def test_timeout_cleanup(tmp_path):
    log = tmp_path / 'timeout.log'
    # A real local process verifies graceful SIGTERM cleanup, not Docker/network.
    code = ('import signal, time, sys; '
            'signal.signal(signal.SIGTERM, lambda *args: (print("cleanup complete"), sys.exit(0))); '
            'print("ready", flush=True); time.sleep(60)')
    result = o.run_command([sys.executable, '-c', code], log, timeout=1)
    assert result.returncode == 124
    assert 'cleanup complete' in result.stdout.splitlines()


def test_evaluator_timeout_is_forwarded_not_killed(tmp_path):
    cmd = o.evaluation_command(dict(run_id='1234abcd', directory=tmp_path), Path('db'), 1.2)
    assert cmd[-2:] == ['--timeout', '2']


def test_launch_failure_is_a_run_error(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('cannot launch')
    monkeypatch.setattr(o, 'run_command', fail)
    result = o.execute(['missing'], tmp_path / 'log', None)
    assert result.returncode == 1 and 'cannot launch' in result.stdout
