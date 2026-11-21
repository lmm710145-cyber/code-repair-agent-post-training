"""Construct reviewed PR tasks. Source review is explicit, never inferred from a diff."""
import subprocess
from pathlib import Path
from ..io import read, write, sha
from ..config import workspace
from .schema import validate_catalog, relative_path


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True).stdout


def build(cfg, spec_path):
    specs = read(spec_path)['tasks']
    out = []
    for spec in specs:
        review = spec['source_review']
        if review.get('status') != 'approved' or not review.get('reviewer') or not review.get('evidence'):
            raise ValueError('Record source evidence, reviewer and approval first')
        if spec.get('source_kind') == 'patch_derived_issue':
            raise ValueError('Patch-derived issues belong in a separate candidate pool')
        iid = relative_path(spec['task_id'])
        if '/' in iid:
            raise ValueError('Invalid task ID')
        repo = (Path(spec_path).resolve().parent / spec['checkout']).resolve()
        base = git(repo, 'rev-parse', spec['base_commit'] + '^{commit}').strip()
        fix = git(repo, 'rev-parse', spec['fix_commit'] + '^{commit}').strip()
        prod, tests = spec['production_paths'], spec['test_paths']
        for path in prod + tests:
            relative_path(path)
        def files(paths):
            return set(git(repo, 'diff', '--name-only', base, fix, '--', *paths).splitlines())
        pf, tf = files(prod), files(tests)
        all_files = set(git(repo, 'diff', '--name-only', base, fix).splitlines())
        if not pf or not tf or pf & tf or pf | tf != all_files:
            raise ValueError('All changed files must be partitioned into nonempty production/test patches')
        public = workspace(cfg, 'tasks', 'public', iid)
        private = workspace(cfg, 'tasks', 'private', iid)
        if public.exists() or private.exists():
            raise FileExistsError('Task already exists: ' + iid)
        public.mkdir(parents=True)
        private.mkdir(parents=True)
        subprocess.run(['git', '-C', str(repo), 'archive', '--format=tar', '-o', str(public/'snapshot.tar'), base], check=True)
        gold = git(repo, 'diff', '--binary', base, fix, '--', *prod)
        hidden = git(repo, 'diff', '--binary', base, fix, '--', *tests)
        (private/'gold.patch').write_text(gold)
        (private/'tests.patch').write_text(hidden)
        task = {k:v for k,v in spec.items() if k not in ('checkout', 'fix_commit')}
        task.update(base_commit=base, snapshot=str((public/'snapshot.tar').relative_to(workspace(cfg))),
                    snapshot_sha256=sha(public/'snapshot.tar'), gold_sha256=sha(private/'gold.patch'),
                    tests_sha256=sha(private/'tests.patch'), production_files=sorted(pf), hidden_test_files=sorted(tf))
        write(private/'source.json', dict(base_commit=base,fix_commit=fix,source_review=review))
        out.append(task)
    validate_catalog(out)
    write(workspace(cfg, cfg['tasks']['catalog']), {'schema_version': 1, 'tasks': out})
    return {'constructed': len(out), 'certification_required': True}
