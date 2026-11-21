"""Shared dependency image + fresh task snapshot. No host mounts or credentials."""
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from ..config import workspace
from ..io import sha, digest
from .bounded import RUNNER
from .protocol import parse_response

SUBMIT = 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'


def docker(*args, timeout=300, check=True):
    return subprocess.run(['docker', *map(str,args)], timeout=timeout, check=check, text=True, capture_output=True)


def image_id(image):
    return docker('image', 'inspect', '--format', '{{.Id}}', image).stdout.strip()


class TaskEnvironment:
    def __init__(self, task, cfg):
        self.task, self.cfg = task, cfg
        self.records = []
        self.name = 'repairlab-' + uuid.uuid4().hex[:12]
        self.started = False
        snapshot = workspace(cfg,task['snapshot'])
        if sha(snapshot) != task['snapshot_sha256']:
            raise ValueError('Snapshot hash mismatch')
        self.actual_image = image_id(task['image'])
        docker('create', '--name', self.name, '--label', 'repairlab.managed=true',
               '--network', 'none', '--cpus', str(cfg['environment']['cpus']),
               '--memory', cfg['environment']['memory'], '--memory-swap', cfg['environment']['memory'],
               '--pids-limit', '256', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--workdir', '/workspace', '--env', 'PYTHONDONTWRITEBYTECODE=1',
               '--env', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1', '--env', 'OMP_NUM_THREADS=1',
               task['image'], 'sleep', 'infinity')
        self.started=True
        try:
            docker('start', self.name)
            docker('cp', snapshot, self.name+':/tmp/snapshot.tar')
            self.shell('mkdir -p /workspace /tmp/agent-repro /opt/repairlab /tmp/repairlab-tests && '
                       'tar -xf /tmp/snapshot.tar -C /workspace && rm /tmp/snapshot.tar && '
                       'git init -q && git config user.email task@example.invalid && '
                       'git config user.name task && git add -A && git commit -qm base', check=True)
            for command in task.get('setup_commands', []):
                self.shell(command, check=True, timeout=300)
            docker('cp', Path(__file__).with_name('observer.py'), self.name+':/opt/repairlab/repairlab_observer.py')
            self.observer_sha = sha(Path(__file__).with_name('observer.py'))
            self.initial_status = self.shell('git status --porcelain=v1',check=True).stdout
            if self.initial_status:
                raise ValueError('Setup changed task workspace; certify a clean setup first')
        except BaseException:
            self.cleanup()
            raise

    def shell(self, command, *, check=False, timeout=120):
        return docker('exec',self.name,'timeout','--kill-after=5',str(timeout),'bash','-lc',command,timeout=timeout+15,check=check)

    def prefix(self, aid):
        env = {'PYTHONPATH':'/opt/repairlab:'+self.task.get('python_path','/workspace/src'),
               'PYTEST_PLUGINS':','.join(['repairlab_observer']+self.task.get('pytest_plugins',[])),
               'PYTEST_ADDOPTS':self.task.get('pytest_addopts',''),
               'REPAIRLAB_TEST_RECORD_DIR':'/tmp/repairlab-tests','REPAIRLAB_ACTION_ID':aid,
               'REPAIRLAB_PRODUCTION_PATHSPEC':json.dumps(self.task['production_paths']),
               'REPAIRLAB_IMPORT_PACKAGE':self.task.get('import_package','')}
        return 'export '+shlex.join([k+'='+v for k,v in env.items()])+'; set -o pipefail; '

    def patch(self, *, full=False):
        self.shell('git add -N -- .',check=True)
        paths = ['.'] if full else self.task['production_paths']+[':(exclude)'+p for p in self.task['test_paths']]+[':(exclude,glob)**/conftest.py',':(exclude,glob)**/tests/**',':(exclude,glob)**/test_*.py']
        return self.shell('git diff --binary HEAD -- '+shlex.join(paths),check=True).stdout

    def state(self):
        # Workspace state, plus common reproduction files; policy never sees this.
        code="import pathlib,hashlib,json; print(json.dumps([(str(p),hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(pathlib.Path('/tmp/agent-repro').rglob('*')) if p.is_file() and not p.is_symlink()]))"
        return digest([self.patch(full=True),self.shell('python -c '+shlex.quote(code),check=True).stdout])

    def execute(self, action):
        parsed = parse_response(action['response'])
        if parsed['command'] != action['command']:
            raise ValueError('Action/parser mismatch')
        command = action['command'];aid=uuid.uuid4().hex;before=self.state()
        spec={'command':self.prefix(aid)+command,'timeout':self.cfg['budget']['command_seconds'],
              'limit':self.cfg['environment']['raw_output_bytes']}
        raw=docker('exec',self.name,'python','-c',RUNNER,json.dumps(spec),timeout=spec['timeout']+20)
        result=json.loads(raw.stdout);output=result['stdout']+result['stderr']
        code="import pathlib,json; print(json.dumps([d for p in pathlib.Path('/tmp/repairlab-tests').glob('*.json') if (d:=json.loads(p.read_text())).get('action_id')=="+repr(aid)+"]))"
        sessions=json.loads(self.shell('python -c '+shlex.quote(code),check=True).stdout)
        observer_ok=self.shell('sha256sum /opt/repairlab/repairlab_observer.py',check=True).stdout.startswith(self.observer_sha)
        e=dict(command=command,before_state=before,after_state=self.state(),output=output,
               output_sha256=hashlib.sha256(output.encode()).hexdigest(),returncode=result['returncode'],
               resource_reason=result['resource_reason'],pytest_sessions=sessions,observer_tampered=not observer_ok)
        self.records.append(e)
        if result['resource_reason']=='raw_output_limit':
            from minisweagent.exceptions import LimitsExceeded
            raise LimitsExceeded({'role':'exit','content':'Raw output resource limit', 'extra':{'exit_status':'RawOutputLimit'}})
        if command.strip()==SUBMIT and result['returncode']==0:
            from minisweagent.exceptions import Submitted
            raise Submitted({'role':'exit','content':'submitted','extra':{'exit_status':'Submitted','submission':self.patch()}})
        summary={'shell_exit':result['returncode'],'pytest':[{'complete':s.get('complete'),'collected':s.get('collected'),'counts':s.get('counts')} for s in sessions], 'observer_tampered':not observer_ok}
        return {'output':output+'\n[EXECUTION EVIDENCE] '+json.dumps(summary),'returncode':result['returncode']}

    def get_template_vars(self):return {'cwd':'/workspace','timeout':self.cfg['budget']['command_seconds']}
    def serialize(self):return {'task_id':self.task['task_id'],'executions':self.records}
    def cleanup(self):
        if self.started:
            docker('rm','-f',self.name,check=False);self.started=False
    def __enter__(self):return self
    def __exit__(self,*args):self.cleanup()
