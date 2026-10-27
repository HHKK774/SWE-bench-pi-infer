"""Offline evaluation regression tests: no Docker or network."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
import evaluate
from run_tracking import migrate_evaluations, record_evaluation


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, 'REPO_ROOT', tmp_path)
    directory = tmp_path / 'trajectories/time__example__issue-1__plain__1234abcd'
    directory.mkdir(parents=True)
    (directory / 'run.json').write_text(json.dumps(dict(instance_id='example__issue-1', model='provider/model')))
    db = tmp_path / 'runs.db'
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.execute('CREATE TABLE runs (run_id TEXT PRIMARY KEY, trajectory_dir TEXT, instance_id TEXT, model TEXT, evaluation TEXT)')
        connection.execute('INSERT INTO runs VALUES (?, ?, ?, ?, NULL)',
                           ('1234abcd', str(directory), 'example__issue-1', 'provider/model'))
    return tmp_path, directory, db


def test_resolution(recorded):
    root, directory, db = recorded
    assert evaluate.resolve_run('1234abcd', db, root / 'absent') == (directory, 'example__issue-1', 'provider/model')
    assert evaluate.resolve_run('1234abcd', root / 'missing.db', Path('trajectories')) == (directory, 'example__issue-1', 'provider/model')
    assert not (root / 'missing.db').exists()
    with pytest.raises(ValueError, match='found 0'):
        evaluate.resolve_run('00000000', db, root / 'trajectories')
    with pytest.raises(ValueError, match='hexadecimal'):
        evaluate.resolve_run('../bad', db, root)


def test_patch_sources(recorded):
    root, directory, _ = recorded
    with pytest.raises(ValueError, match='--patch FILE'):
        evaluate.load_prediction(directory, 'example__issue-1', 'provider/model')
    prediction = dict(instance_id='example__issue-1', model_name_or_path='provider/model', model_patch='old\n')
    fallback = directory.parent / 'predictions.jsonl'
    fallback.write_text(json.dumps(prediction) + '\n')
    assert evaluate.load_prediction(directory, 'example__issue-1', None) == prediction
    (directory / 'patch.diff').write_bytes(b'')
    assert evaluate.load_prediction(directory, 'example__issue-1', 'provider/model')['model_patch'] == ''
    override = root / 'override.diff'
    override.write_bytes('diff\r\n+é\n'.encode())
    assert evaluate.load_prediction(directory, 'example__issue-1', 'provider/model', override)['model_patch'] == 'diff\r\n+é\n'
    (directory / 'patch.diff').unlink()
    fallback.write_text((json.dumps(prediction) + '\n') * 2)
    with pytest.raises(ValueError, match='ambiguous'):
        evaluate.load_prediction(directory, 'example__issue-1', 'provider/model')


def report_file(root, verdict=True):
    path = root / 'logs/evaluation/1234abcd/provider__model/example__issue-1/report.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'example__issue-1': dict(resolved=verdict, patch_successfully_applied=True, tests_status={})}))
    return path


def test_report_and_database(recorded):
    root, _, db = recorded
    path = report_file(root)
    result = evaluate.parse_report(path, 'example__issue-1', '1234abcd')
    assert set(result) == {'resolved', 'patch_successfully_applied', 'tests_status', 'eval_run_id', 'report', 'evaluated_at'}
    assert result['report'] == str(path.relative_to(root))
    assert result['evaluated_at'].endswith('+00:00')
    for verdict in ('success', 'fail'):
        record_evaluation(db, '1234abcd', verdict)
        with closing(sqlite3.connect(db)) as connection:
            assert connection.execute('SELECT evaluation FROM runs').fetchone()[0] == verdict
    with pytest.raises(ValueError, match='absent'):
        record_evaluation(db, '00000000', 'success')
    for invalid in (None, True, {}, [], 'SUCCESS', '', '{"resolved": true}'):
        with pytest.raises(ValueError, match='verdict'):
            record_evaluation(db, '1234abcd', invalid)
    path.write_text('{"resolved": "false"}')
    with pytest.raises(ValueError, match='Invalid resolved'):
        evaluate.parse_report(path, 'example__issue-1', '1234abcd')


@pytest.mark.parametrize('cli_exit,resolved', [(0, True), (0, False), (7, False)])
def test_full_flow(recorded, monkeypatch, capsys, cli_exit, resolved):
    root, directory, db = recorded
    (directory / 'patch.diff').write_text('diff\n')
    bundle = root / 'bundle'
    bundle.mkdir()
    (bundle / 'instance.json').write_text('[{"instance_id":"example__issue-1"}]')
    paths = []
    def run(command, cwd):
        assert cwd == root
        assert command[1:3] == ['-m', 'swebench.harness.run_evaluation']
        assert command[command.index('--run_id') + 1] == '1234abcd'
        path = Path(command[command.index('--predictions_path') + 1])
        paths.append(path)
        assert len(path.read_text().splitlines()) == 1
        assert json.loads(path.read_text())['model_patch'] == 'diff\n'
        report_file(root, resolved)
        return SimpleNamespace(returncode=cli_exit)
    monkeypatch.setattr(evaluate.subprocess, 'run', run)
    assert evaluate.main(['1234abcd', '--runs-db', str(db), '--bundle-dir', str(bundle)]) == int(cli_exit != 0)
    summary = json.loads(capsys.readouterr().out)
    assert summary['evaluation_stored'] is True
    assert summary['resolved'] is resolved
    assert summary['verdict'] == ('success' if resolved else 'fail')
    assert not paths[0].exists()
    with closing(sqlite3.connect(db)) as connection:
        assert connection.execute("SELECT evaluation FROM runs WHERE run_id='1234abcd'").fetchone()[0] == ('success' if resolved else 'fail')


@pytest.mark.parametrize('mode', ['missing', 'malformed', 'missing-run'])
def test_failed_evaluation(recorded, monkeypatch, capsys, mode):
    root, directory, db = recorded
    (directory / 'patch.diff').write_text('diff\n')
    bundle = root / 'bundle'
    bundle.mkdir()
    (bundle / 'instance.json').write_text('[{"instance_id":"example__issue-1"}]')
    stale = report_file(root)
    if mode == 'missing-run':
        with closing(sqlite3.connect(db)) as connection, connection:
            connection.execute('DELETE FROM runs')
    def run(command, cwd):
        assert not stale.exists()
        if mode == 'malformed':
            report_file(root).write_text('not JSON')
        elif mode == 'missing-run':
            report_file(root)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(evaluate.subprocess, 'run', run)
    assert evaluate.main(['1234abcd', '--runs-db', str(db), '--bundle-dir', str(bundle)]) == 1
    assert json.loads(capsys.readouterr().out)['evaluation_stored'] is False


def test_default_fetch_and_keep_predictions(recorded, monkeypatch, capsys):
    root, directory, db = recorded
    home = root / 'home'
    cache = home / '.cache/swebench'
    cache.mkdir(parents=True)
    (cache / 'instances.db').write_bytes(b'cache sentinel')
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: home))
    (directory / 'patch.diff').write_bytes(b'')
    commands, predictions = [], []
    def run(command, cwd, **kwargs):
        commands.append(command)
        if 'fetch_instance.py' in command[1]:
            assert '--refresh' not in command
            bundle = Path(command[3])
            bundle.mkdir()
            (bundle / 'instance.json').write_text('[{"instance_id":"example__issue-1"}]')
        else:
            predictions.append(Path(command[command.index('--predictions_path') + 1]))
            report_file(root, False)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(evaluate.subprocess, 'run', run)
    try:
        assert evaluate.main(['1234abcd', '--runs-db', str(db), '--keep-predictions']) == 0
        assert len(commands) == 2
        assert predictions[0].is_file()
        assert (cache / 'instances.db').read_bytes() == b'cache sentinel'
        assert 'Predictions retained:' in capsys.readouterr().err
    finally:
        for path in predictions:
            path.unlink(missing_ok=True)


def test_migration(recorded, monkeypatch, capsys):
    _, _, db = recorded
    values = {
        'old-true': '{"resolved": true, "tests_status": {}}',
        'old-false': '{"resolved": false}',
        'done-success': 'success', 'done-fail': 'fail',
        'broken': 'not JSON', 'missing': '{}', 'non-object': '[]',
    }
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.executemany('INSERT INTO runs (run_id, evaluation) VALUES (?, ?)', values.items())
    def forbidden(*args, **kwargs):
        pytest.fail('Migration must not invoke subprocesses')
    monkeypatch.setattr(evaluate.subprocess, 'run', forbidden)
    assert evaluate.main(['--migrate', '--runs-db', str(db)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {'migrated': 2}
    for run_id in ('broken', 'missing', 'non-object'):
        assert run_id in captured.err
    expected = dict(values, **{'old-true': 'success', 'old-false': 'fail', '1234abcd': None})
    with closing(sqlite3.connect(db)) as connection:
        assert dict(connection.execute('SELECT run_id, evaluation FROM runs')) == expected
    assert migrate_evaluations(db) == 0
    with closing(sqlite3.connect(db)) as connection:
        assert dict(connection.execute('SELECT run_id, evaluation FROM runs')) == expected


def test_setup_error(capsys):
    assert evaluate.main(['bad']) == 1
    assert 'hexadecimal' in capsys.readouterr().err
