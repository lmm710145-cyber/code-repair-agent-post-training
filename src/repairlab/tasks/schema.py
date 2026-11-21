from pathlib import Path, PurePosixPath
from ..io import read

DEFECT_TYPES = ('api_contract', 'boundary_defaults', 'parsing_encoding', 'validation_errors',
                'resource_lifecycle', 'state_concurrency', 'semantic_transform', 'rendering_layout')


def relative_path(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or '..' in p.parts or value.startswith('-'):
        raise ValueError(f'Expected safe relative path: {value}')
    return value


def validate_catalog(tasks):
    ids = set()
    repos = {}
    for task in tasks:
        required = ('task_id', 'repo', 'split', 'defect_type', 'problem_statement', 'image',
                    'snapshot', 'snapshot_sha256', 'production_paths', 'test_paths', 'source_review')
        if any(not task.get(k) for k in required):
            raise ValueError('Task is missing required fields')
        relative_path(task['task_id'])
        if '/' in task['task_id'] or task['task_id'] in ids:
            raise ValueError('Task IDs must be unique simple names')
        ids.add(task['task_id'])
        if task['split'] not in ('train', 'dev', 'test') or task['defect_type'] not in DEFECT_TYPES:
            raise ValueError('Invalid split or defect type')
        if task['repo'] in repos and repos[task['repo']] != task['split']:
            raise ValueError('Repository leakage across train/dev/test: ' + task['repo'])
        repos[task['repo']] = task['split']
        if task['source_review'].get('status') != 'approved':
            raise ValueError('Source review must be approved before task use')
        for p in task['production_paths'] + task['test_paths']:
            relative_path(p)
    return tasks


def catalog(cfg):
    p = Path(cfg['workspace']) / cfg['tasks']['catalog']
    data = read(p)
    return validate_catalog(data['tasks'])
