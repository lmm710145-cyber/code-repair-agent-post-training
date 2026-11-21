"""Fresh-container hidden F2P/P2P and broader regression checks."""
import shlex
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from ..agent.environment import TaskEnvironment, docker, image_id
from ..config import workspace
from ..io import write, read, sha, digest


def junit(text):
    root=ET.fromstring(text);states={}
    for case in root.iter('testcase'):
        key=case.get('classname','')+'::'+case.get('name','')
        if key in states:raise ValueError('Duplicate JUnit test identity: '+key)
        states[key]=next((k for k in ('failure','error','skipped') if case.find(k) is not None),'passed')
    if not states:raise ValueError('Zero collected tests')
    return states


def run_suite(task,cfg,patch,files):
    private=workspace(cfg,'tasks','private',task['task_id'])
    with TaskEnvironment(task,cfg) as env, tempfile.TemporaryDirectory() as temp:
        if patch:
            p=Path(temp)/'candidate.patch';p.write_text(patch)
            docker('cp',p,env.name+':/tmp/candidate.patch')
            r=env.shell('git apply --check /tmp/candidate.patch && git apply /tmp/candidate.patch')
            if r.returncode:return {'valid':False,'reason':'patch_apply','states':{}}
        if sha(private/'tests.patch')!=task['tests_sha256']:raise ValueError('Hidden test patch changed')
        docker('cp',private/'tests.patch',env.name+':/tmp/tests.patch')
        r=env.shell('git apply --check /tmp/tests.patch && git apply /tmp/tests.patch')
        if r.returncode:return {'valid':False,'reason':'hidden_patch_apply','states':{}}
        command=env.prefix('grader')+'python -m pytest -q -o addopts= --junitxml=/tmp/result.xml '+shlex.join(task.get('pytest_args',[])+files)
        r=env.shell(command,timeout=cfg['environment']['grading_seconds'])
        xml=env.shell('cat /tmp/result.xml')
        try:states=junit(xml.stdout)
        except (ET.ParseError,ValueError):return {'valid':False,'reason':'invalid_or_empty_junit','states':{},'returncode':r.returncode}
        return {'valid':r.returncode in (0,1) and 'error' not in states.values(),
                'states':states,'returncode':r.returncode,'output':r.stdout+r.stderr}


def certify(task,cfg):
    private=workspace(cfg,'tasks','private',task['task_id'])
    if sha(private/'gold.patch')!=task['gold_sha256']:raise ValueError('Gold hash mismatch')
    gold=(private/'gold.patch').read_text();runs=[]
    for _ in range(cfg['tasks']['certification_repeats']):
        base=run_suite(task,cfg,'',task['hidden_test_files'])
        fixed=run_suite(task,cfg,gold,task['hidden_test_files'])
        broad_base=run_suite(task,cfg,'',task['regression_tests'])
        broad_fixed=run_suite(task,cfg,gold,task['regression_tests'])
        if not all(r['valid'] for r in [base,fixed,broad_base,broad_fixed]):raise ValueError('Invalid certification test execution')
        f2p=sorted(k for k,v in base['states'].items() if v=='failure' and fixed['states'].get(k)=='passed')
        p2p=sorted(k for k,v in base['states'].items() if v=='passed')
        regress=sorted(k for k,v in broad_base['states'].items() if v=='passed')
        known=sorted(k for k,v in broad_base['states'].items() if v in ('failure','error'))
        if not f2p or not p2p or fixed['returncode']!=0:raise ValueError('Need nonempty F2P/P2P and clean gold oracle')
        if any(fixed['states'].get(k)!='passed' for k in f2p+p2p):raise ValueError('Gold failed F2P/P2P')
        if any(broad_fixed['states'].get(k)!='passed' for k in regress):raise ValueError('Gold regressed')
        if any(v in ('failure','error') and k not in known for k,v in broad_fixed['states'].items()):raise ValueError('Gold introduced new broader failure')
        runs.append(dict(f2p=f2p,p2p=p2p,regression_required=regress,known_baseline_failures=known))
    if any(r!=runs[0] for r in runs):raise ValueError('Nondeterministic task certification')
    result=dict(runs[0],certified=True,task_digest=digest(task),image_id=image_id(task['image']),repeats=len(runs))
    write(private/'certification.json',result)
    return result


def grade(task,cfg,patch,folder):
    private=workspace(cfg,'tasks','private',task['task_id']);cert=read(private/'certification.json')
    if not cert['certified'] or cert['task_digest']!=digest(task) or cert['image_id']!=image_id(task['image']):
        raise ValueError('Task/image certification changed')
    oracle=run_suite(task,cfg,patch,task['hidden_test_files'])
    broad=run_suite(task,cfg,patch,task['regression_tests'])
    regression=any(broad['states'].get(k)!='passed' for k in cert['regression_required'])
    new_fail=[k for k,v in broad['states'].items() if v in ('failure','error') and k not in cert['known_baseline_failures']]
    resolved=bool(patch.strip()) and oracle['valid'] and broad['valid'] and oracle['returncode']==0 and not regression and not new_fail and all(oracle['states'].get(k)=='passed' for k in cert['f2p']+cert['p2p'])
    result=dict(resolved=resolved,regression=regression or bool(new_fail),oracle=oracle,broader=broad,new_failures=new_fail)
    write(Path(folder)/'grader.json',result)
    return result
