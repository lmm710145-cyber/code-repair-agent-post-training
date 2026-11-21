"""Small immutable-run and JSON utilities; no machine-specific paths."""
import contextlib
import fcntl
import hashlib
import json
import os
import uuid
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


@contextlib.contextmanager
def lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def freeze(folder, value):
    value = dict(value, runtime_sources=runtime_sources())
    path = Path(folder) / 'run-config.json'
    if path.exists() and read(path) != value:
        raise ValueError('Run inputs changed; use a new output directory')
    write(path, value)


def runtime_sources():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*.py'))}


def verify_run(folder):
    expected = read(Path(folder) / 'run-config.json')['runtime_sources']
    if expected != runtime_sources():
        raise ValueError('Runtime source changed; create a new run')
